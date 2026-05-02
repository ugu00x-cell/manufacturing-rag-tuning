"""リランキングモジュール.

手順:
    1. baseline と同じ方法で Top10 を取得
    2. cross-encoder/ms-marco-MiniLM-L-6-v2 でリランキング
    3. リランキング後 Top3 を最終結果に使う
    4. evaluate.py で精度比較

実行:
    python -m src.reranker
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol

from src.baseline import (
    ChromaIndex,
    Document,
    Embedder,
    OllamaClient,
    Question,
    SBertEmbedder,
    SYSTEM_PROMPT,
    build_index,
    build_user_prompt,
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

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_INITIAL_TOP_K = 10  # ベクトル検索で取る件数
FINAL_TOP_K = 3            # リランキング後の最終件数


# ============================================================
# Reranker プロトコル（テスト時にモック可能）
# ============================================================
class Reranker(Protocol):
    """リランカープロトコル."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """質問×候補テキストのスコア（高いほど関連性大）を返す."""
        ...


class CrossEncoderReranker:
    """sentence-transformers の CrossEncoder ラッパ（実機用）."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        """初期化（モデル読み込みは初回 rerank 時に遅延）.

        Args:
            model_name: HuggingFace モデル名
        """
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        """モデルを遅延ロード."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Reranker model 読み込み: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """各候補に対するスコアを返す.

        Args:
            query: 質問テキスト
            candidates: 候補テキスト list

        Returns:
            候補と同順のスコア list（高いほど関連性大）
        """
        self._load()
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


# ============================================================
# パイプライン
# ============================================================
def retrieve_with_rerank(
    question: str,
    index: ChromaIndex,
    embedder: Embedder,
    reranker: Reranker,
    initial_top_k: int = RERANK_INITIAL_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> list[tuple[str, str]]:
    """ベクトル検索 → リランキング → 最終 Top-K.

    Args:
        question: 質問
        index: ベクトル検索インデックス
        embedder: Embedder
        reranker: Reranker
        initial_top_k: ベクトル検索段階の取得件数
        final_top_k: リランキング後の最終件数

    Returns:
        [(doc_id, chunk_text), ...]（最終 Top-K・doc単位重複除去後）
    """
    # 1. ベクトル検索（多めに取得）
    raw = index.query(embedder.embed([question])[0], top_k=initial_top_k * 2)

    # doc_id 単位で重複除去
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for chunk_id, chunk_text_, _ in raw:
        doc_id = chunk_id_to_doc_id(chunk_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        candidates.append((doc_id, chunk_text_))
        if len(candidates) >= initial_top_k:
            break

    if not candidates:
        return []

    # 2. リランキング
    texts = [c[1] for c in candidates]
    scores = reranker.rerank(question, texts)

    # スコア降順で並べ替え
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:final_top_k]]


def run_rerank_evaluation(
    documents: list[Document],
    questions: list[Question],
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    initial_top_k: int = RERANK_INITIAL_TOP_K,
    final_top_k: int = FINAL_TOP_K,
    save_csv: bool = True,
) -> tuple[dict, list[QueryResult]]:
    """全質問に対してリランキング込みで評価する.

    Args:
        documents: ナレッジベース
        questions: 評価質問
        embedder: Embedder（None なら SBertEmbedder）
        reranker: Reranker（None なら CrossEncoderReranker）
        initial_top_k: ベクトル検索段階の取得件数
        final_top_k: 最終 Top-K
        save_csv: 結果を CSV 保存するか

    Returns:
        (サマリー, 各質問の QueryResult)
    """
    embedder = embedder or SBertEmbedder()
    reranker = reranker or CrossEncoderReranker()

    index = build_index(documents, embedder=embedder, collection_name="reranker_idx")

    query_results: list[QueryResult] = []
    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve_with_rerank(
            q["question"],
            index=index,
            embedder=embedder,
            reranker=reranker,
            initial_top_k=initial_top_k,
            final_top_k=final_top_k,
        )
        elapsed = time.perf_counter() - start

        query_results.append(QueryResult(
            question_id=q["id"],
            retrieved_doc_ids=[d for d, _ in retrieved],
            relevant_doc_ids=q["relevant_doc_ids"],
            elapsed_sec=round(elapsed, 4),
        ))

    summary = evaluate_results(method="reranker", query_results=query_results, k=final_top_k)
    print_summary(summary)
    if save_csv:
        save_results_csv("reranker", summary, query_results)
    return dict(summary), query_results


def main() -> None:
    """エントリポイント."""
    documents, questions = load_data()
    run_rerank_evaluation(documents, questions)


if __name__ == "__main__":
    main()
