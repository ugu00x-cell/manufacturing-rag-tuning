"""ハイブリッド検索モジュール.

ベクトル検索（密ベクトル）と BM25（疎ベクトル）を組み合わせて検索する。
スコア統合: RRF (Reciprocal Rank Fusion)

実行:
    python -m src.hybrid_search
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Protocol

from src.baseline import (
    ChromaIndex,
    Document,
    Embedder,
    Question,
    SBertEmbedder,
    build_index,
    chunk_documents,
    chunk_id_to_doc_id,
    load_data,
)
from src.evaluate import (
    QueryResult,
    evaluate_results,
    print_summary,
    save_results_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VECTOR_TOP_K = 10  # ベクトル検索段階の取得件数
BM25_TOP_K = 10    # BM25 段階の取得件数
FINAL_TOP_K = 3    # RRF 後の最終件数
RRF_K = 60         # RRF 定数（標準値60）


# ============================================================
# 日本語トークナイザ（最小限）
# ============================================================
def simple_tokenize(text: str) -> list[str]:
    """シンプルな日本語混じりテキストのトークン化.

    アルファベット・数字は単語単位、CJK文字は1文字ずつ（unigram）として扱う。
    BM25 の語彙単位として粗いが、形態素解析器を入れずに済む。

    Args:
        text: 入力テキスト

    Returns:
        トークン list
    """
    tokens: list[str] = []
    # 英数字を抽出（連続）
    for m in re.finditer(r"[A-Za-z0-9]+", text):
        tokens.append(m.group(0).lower())
    # CJK 文字を1文字ずつ
    for ch in re.findall(r"[一-龥ぁ-んァ-ヶ]", text):
        tokens.append(ch)
    return tokens


# ============================================================
# BM25 検索プロトコル（テスト時にモック可能）
# ============================================================
class BM25Engine(Protocol):
    """BM25 検索エンジンプロトコル."""

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """[(corpus index, score), ...] を返す（高スコア順）."""
        ...


class RankBM25Engine:
    """rank_bm25 ライブラリベースの実装."""

    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        """初期化.

        Args:
            corpus_tokens: 各文書のトークンlist
        """
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 検索を実行.

        Args:
            query: 質問
            top_k: 取得件数

        Returns:
            [(corpus index, score), ...]（スコア降順）
        """
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # numpy 配列でも list でも受け付ける
        idx_scores = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [(int(i), float(s)) for i, s in idx_scores[:top_k]]


# ============================================================
# RRF
# ============================================================
def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """複数のランキングを RRF で統合する.

    Args:
        rankings: 各検索手法のランキング（doc_id を順位順に並べた list）
        k: RRF 定数

    Returns:
        [(doc_id, fused_score), ...]（融合スコア降順）
    """
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# パイプライン
# ============================================================
def hybrid_retrieve(
    question: str,
    index: ChromaIndex,
    embedder: Embedder,
    bm25: BM25Engine,
    chunk_ids: list[str],
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> list[str]:
    """ハイブリッド検索を実行する.

    Args:
        question: 質問
        index: ベクトル検索インデックス
        embedder: Embedder
        bm25: BM25 エンジン
        chunk_ids: BM25コーパスと同順の chunk_id list（index→chunk_id 解決用）
        vector_top_k: ベクトル検索段階
        bm25_top_k: BM25 段階
        final_top_k: 最終件数

    Returns:
        doc_id list（最終 Top-K・重複除去後）
    """
    # ベクトル検索
    vec_raw = index.query(embedder.embed([question])[0], top_k=vector_top_k)
    vec_doc_ids: list[str] = []
    seen_v: set[str] = set()
    for chunk_id, _, _ in vec_raw:
        d = chunk_id_to_doc_id(chunk_id)
        if d not in seen_v:
            seen_v.add(d)
            vec_doc_ids.append(d)

    # BM25 検索
    bm25_results = bm25.search(question, top_k=bm25_top_k)
    bm25_doc_ids: list[str] = []
    seen_b: set[str] = set()
    for idx, _ in bm25_results:
        d = chunk_id_to_doc_id(chunk_ids[idx])
        if d not in seen_b:
            seen_b.add(d)
            bm25_doc_ids.append(d)

    # RRF 融合
    fused = reciprocal_rank_fusion([vec_doc_ids, bm25_doc_ids])
    return [doc_id for doc_id, _ in fused[:final_top_k]]


def build_bm25_corpus(documents: list[Document]) -> tuple[BM25Engine, list[str]]:
    """ナレッジベースから BM25 コーパスを構築する.

    Args:
        documents: ナレッジベース

    Returns:
        (BM25Engine, chunk_ids)
        chunk_ids は corpus index と同順
    """
    chunks = chunk_documents(documents)  # baseline と同じチャンク設定
    chunk_ids: list[str] = []
    corpus_tokens: list[list[str]] = []
    for i, (doc_id, chunk) in enumerate(chunks):
        chunk_ids.append(f"{doc_id}__chunk{i:03d}")
        corpus_tokens.append(simple_tokenize(chunk))

    engine = RankBM25Engine(corpus_tokens)
    logger.info("BM25 コーパス構築完了: %d チャンク", len(chunk_ids))
    return engine, chunk_ids


def run_hybrid_evaluation(
    documents: list[Document],
    questions: list[Question],
    embedder: Embedder | None = None,
    bm25: BM25Engine | None = None,
    save_csv: bool = True,
) -> tuple[dict, list[QueryResult]]:
    """全質問に対してハイブリッド検索を評価する.

    Args:
        documents: ナレッジベース
        questions: 評価質問
        embedder: Embedder（None なら SBertEmbedder）
        bm25: BM25Engine（None なら自動構築）
        save_csv: CSV保存するか

    Returns:
        (サマリー, 各質問の QueryResult)
    """
    embedder = embedder or SBertEmbedder()
    index = build_index(documents, embedder=embedder, collection_name="hybrid_idx")

    if bm25 is None:
        bm25, chunk_ids = build_bm25_corpus(documents)
    else:
        # 外部から bm25 を渡す場合、chunk_ids も自前で持っている前提。テスト用パス。
        chunks = chunk_documents(documents)
        chunk_ids = [f"{doc_id}__chunk{i:03d}" for i, (doc_id, _) in enumerate(chunks)]

    query_results: list[QueryResult] = []
    for q in questions:
        start = time.perf_counter()
        retrieved_doc_ids = hybrid_retrieve(
            q["question"],
            index=index,
            embedder=embedder,
            bm25=bm25,
            chunk_ids=chunk_ids,
        )
        elapsed = time.perf_counter() - start

        query_results.append(QueryResult(
            question_id=q["id"],
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=q["relevant_doc_ids"],
            elapsed_sec=round(elapsed, 4),
        ))

    summary = evaluate_results(method="hybrid_search", query_results=query_results, k=FINAL_TOP_K)
    print_summary(summary)
    if save_csv:
        save_results_csv("hybrid_search", summary, query_results)
    return dict(summary), query_results


def main() -> None:
    """エントリポイント."""
    documents, questions = load_data()
    run_hybrid_evaluation(documents, questions)


if __name__ == "__main__":
    main()
