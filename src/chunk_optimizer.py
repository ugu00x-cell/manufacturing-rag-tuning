"""チャンクサイズ最適化モジュール.

5パターンのチャンク設定を比較し、Recall@3 / MRR / 応答時間を CSV 保存する。

実行:
    python -m src.chunk_optimizer
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import TypedDict

from src.baseline import (
    Document,
    Embedder,
    Question,
    SBertEmbedder,
    build_index,
    chunk_id_to_doc_id,
    load_data,
)
from src.evaluate import (
    QueryResult,
    evaluate_results,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class ChunkPattern(TypedDict):
    """チャンク設定."""

    name: str
    chunk_size: int
    overlap: int


# 仕様準拠の5パターン
PATTERNS: list[ChunkPattern] = [
    {"name": "200_no_overlap", "chunk_size": 200, "overlap": 0},
    {"name": "200_overlap50", "chunk_size": 200, "overlap": 50},
    {"name": "500_no_overlap", "chunk_size": 500, "overlap": 0},
    {"name": "500_overlap100", "chunk_size": 500, "overlap": 100},
    {"name": "1000_overlap200", "chunk_size": 1000, "overlap": 200},
]


def evaluate_pattern(
    pattern: ChunkPattern,
    documents: list[Document],
    questions: list[Question],
    embedder: Embedder,
    top_k: int = 3,
) -> tuple[dict, list[QueryResult]]:
    """1パターンを評価する.

    Args:
        pattern: ChunkPattern
        documents: ナレッジベース
        questions: 評価質問
        embedder: 埋め込みベクトル化器
        top_k: Top-K

    Returns:
        (サマリー辞書, 各質問の検索結果)
    """
    logger.info("=== パターン: %s ===", pattern["name"])
    index = build_index(
        documents,
        embedder=embedder,
        chunk_size=pattern["chunk_size"],
        overlap=pattern["overlap"],
        collection_name=f"chunk_{pattern['name']}",
    )

    query_results: list[QueryResult] = []
    for q in questions:
        start = time.perf_counter()
        raw = index.query(embedder.embed([q["question"]])[0], top_k=top_k * 3)
        elapsed = time.perf_counter() - start

        # doc_id 単位で重複除去
        seen: set[str] = set()
        retrieved_doc_ids: list[str] = []
        for chunk_id, _, _ in raw:
            doc_id = chunk_id_to_doc_id(chunk_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            retrieved_doc_ids.append(doc_id)
            if len(retrieved_doc_ids) >= top_k:
                break

        query_results.append(QueryResult(
            question_id=q["id"],
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=q["relevant_doc_ids"],
            elapsed_sec=round(elapsed, 4),
        ))

    summary = evaluate_results(method=f"chunk_{pattern['name']}", query_results=query_results, k=top_k)
    print_summary(summary)
    return dict(summary), query_results


def run_all_patterns(
    documents: list[Document],
    questions: list[Question],
    embedder: Embedder | None = None,
    top_k: int = 3,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """全パターンを評価し、CSV にまとめて保存する.

    Args:
        documents: ナレッジベース
        questions: 評価質問
        embedder: Embedder（None なら SBertEmbedder を生成）
        top_k: Top-K
        results_dir: 出力先

    Returns:
        保存した CSV のパス
    """
    if embedder is None:
        embedder = SBertEmbedder()

    summaries: list[dict] = []
    for pattern in PATTERNS:
        summary, _ = evaluate_pattern(pattern, documents, questions, embedder, top_k=top_k)
        summary["chunk_size"] = pattern["chunk_size"]
        summary["overlap"] = pattern["overlap"]
        summaries.append(summary)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "chunk_optimizer_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "chunk_size", "overlap", "n_questions",
                        "recall_at_k", "mrr", "mean_elapsed_sec", "k"],
        )
        writer.writeheader()
        writer.writerows(summaries)

    logger.info("チャンク比較サマリーを保存: %s", out_path)
    return out_path


def main() -> None:
    """エントリポイント."""
    documents, questions = load_data()
    run_all_patterns(documents, questions)


if __name__ == "__main__":
    main()
