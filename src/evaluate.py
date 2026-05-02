"""精度評価モジュール（全手法共通）.

メトリクス:
    - Recall@K   : 正解ドキュメントが Top-K に含まれる割合
    - MRR        : 正解の最高順位の逆数の平均（Mean Reciprocal Rank）
    - 応答時間   : 検索〜生成の合計秒数（平均）

出力:
    - コンソール表示
    - results/<method>_<timestamp>.csv
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class QueryResult(TypedDict):
    """1質問の検索＋応答結果."""

    question_id: str
    retrieved_doc_ids: list[str]   # 検索Top-K（順位順）
    relevant_doc_ids: list[str]    # 正解
    elapsed_sec: float


class EvalSummary(TypedDict):
    """評価サマリー."""

    method: str
    n_questions: int
    recall_at_k: float
    mrr: float
    mean_elapsed_sec: float
    k: int


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall@K を計算する.

    Args:
        retrieved: 検索結果の doc_id list（順位順）
        relevant: 正解 doc_id list
        k: Top-K

    Returns:
        正解のうち Top-K に含まれる割合（0.0〜1.0）
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hit = sum(1 for d in relevant if d in top_k)
    return hit / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1質問の Reciprocal Rank を計算する.

    Args:
        retrieved: 検索結果の doc_id list（順位順）
        relevant: 正解 doc_id list

    Returns:
        最初に正解が出る順位の逆数（見つからない場合は 0.0）
    """
    rel_set = set(relevant)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel_set:
            return 1.0 / i
    return 0.0


def evaluate_results(
    method: str,
    query_results: list[QueryResult],
    k: int = 3,
) -> EvalSummary:
    """検索結果群から評価サマリーを計算する.

    Args:
        method: 手法名（baseline / reranker / hybrid_search 等）
        query_results: 各質問の検索結果
        k: Top-K（既定3）

    Returns:
        EvalSummary
    """
    if not query_results:
        return EvalSummary(
            method=method,
            n_questions=0,
            recall_at_k=0.0,
            mrr=0.0,
            mean_elapsed_sec=0.0,
            k=k,
        )

    recalls = [
        recall_at_k(r["retrieved_doc_ids"], r["relevant_doc_ids"], k)
        for r in query_results
    ]
    rrs = [
        reciprocal_rank(r["retrieved_doc_ids"], r["relevant_doc_ids"])
        for r in query_results
    ]
    elapseds = [r["elapsed_sec"] for r in query_results]

    return EvalSummary(
        method=method,
        n_questions=len(query_results),
        recall_at_k=round(sum(recalls) / len(recalls), 4),
        mrr=round(sum(rrs) / len(rrs), 4),
        mean_elapsed_sec=round(sum(elapseds) / len(elapseds), 3),
        k=k,
    )


def print_summary(summary: EvalSummary) -> None:
    """サマリーをコンソール表示する.

    Args:
        summary: evaluate_results の戻り値
    """
    print()
    print("=" * 60)
    print(f"  手法: {summary['method']}")
    print("=" * 60)
    print(f"  質問数         : {summary['n_questions']}")
    print(f"  Recall@{summary['k']}      : {summary['recall_at_k']:.4f}")
    print(f"  MRR           : {summary['mrr']:.4f}")
    print(f"  平均応答時間   : {summary['mean_elapsed_sec']:.3f} 秒")
    print("=" * 60)


def save_results_csv(
    method: str,
    summary: EvalSummary,
    query_results: list[QueryResult],
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """検索結果と評価サマリーを CSV 保存する.

    Args:
        method: 手法名
        summary: evaluate_results の戻り値
        query_results: 各質問の検索結果
        results_dir: 出力先ディレクトリ

    Returns:
        書き出した CSV のパス
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{method}_{timestamp}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # ヘッダー：サマリー
        writer.writerow(["# Method", summary["method"]])
        writer.writerow(["# N_questions", summary["n_questions"]])
        writer.writerow([f"# Recall@{summary['k']}", summary["recall_at_k"]])
        writer.writerow(["# MRR", summary["mrr"]])
        writer.writerow(["# Mean_elapsed_sec", summary["mean_elapsed_sec"]])
        writer.writerow([])

        # 各質問の結果
        writer.writerow(
            ["question_id", "retrieved", "relevant", "recall", "rr", "elapsed_sec"]
        )
        for r in query_results:
            writer.writerow([
                r["question_id"],
                "|".join(r["retrieved_doc_ids"]),
                "|".join(r["relevant_doc_ids"]),
                round(recall_at_k(r["retrieved_doc_ids"], r["relevant_doc_ids"], summary["k"]), 4),
                round(reciprocal_rank(r["retrieved_doc_ids"], r["relevant_doc_ids"]), 4),
                r["elapsed_sec"],
            ])

    logger.info("評価結果を保存: %s", out_path)
    return out_path
