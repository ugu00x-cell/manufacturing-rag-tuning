"""evaluate.py のテスト."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluate import (
    QueryResult,
    evaluate_results,
    print_summary,
    recall_at_k,
    reciprocal_rank,
    save_results_csv,
)


# ============================================================
# 正常系
# ============================================================
class TestNormal:
    """正常系."""

    def test_recall_at_k_full_hit(self) -> None:
        """全正解が Top-K に含まれる場合 Recall=1.0."""
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_reciprocal_rank_first_hit(self) -> None:
        """最初のヒットが順位1なら RR=1.0."""
        assert reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_reciprocal_rank_third_hit(self) -> None:
        """3位でヒットなら RR=1/3."""
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3)


# ============================================================
# 異常系
# ============================================================
class TestAbnormal:
    """異常系."""

    def test_recall_no_relevant_returns_zero(self) -> None:
        """正解が空なら Recall=0."""
        assert recall_at_k(["a", "b"], [], k=3) == 0.0

    def test_reciprocal_rank_no_match_returns_zero(self) -> None:
        """ヒットなしなら RR=0."""
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_evaluate_empty_results(self) -> None:
        """空入力でも evaluate_results は壊れない."""
        s = evaluate_results("test", [], k=3)
        assert s["n_questions"] == 0
        assert s["recall_at_k"] == 0.0


# ============================================================
# 境界値
# ============================================================
class TestBoundary:
    """境界値."""

    def test_recall_partial_hit(self) -> None:
        """部分的にヒット = 0.5."""
        # 正解2件のうち1件だけ Top-3 に含まれる
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


# ============================================================
# evaluate_results
# ============================================================
def test_evaluate_results_aggregates_correctly() -> None:
    """サマリーが期待値で集計されること."""
    qr = [
        QueryResult(
            question_id="q1",
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids=["d1"],
            elapsed_sec=1.0,
        ),
        QueryResult(
            question_id="q2",
            retrieved_doc_ids=["x", "y", "z"],  # ヒットなし
            relevant_doc_ids=["d99"],
            elapsed_sec=2.0,
        ),
    ]
    s = evaluate_results("baseline", qr, k=3)
    assert s["n_questions"] == 2
    assert s["recall_at_k"] == pytest.approx(0.5)
    assert s["mrr"] == pytest.approx(0.5)
    assert s["mean_elapsed_sec"] == 1.5


def test_save_results_csv_writes_file(tmp_path: Path) -> None:
    """CSVが書き出されること."""
    qr = [
        QueryResult(
            question_id="q1",
            retrieved_doc_ids=["d1"],
            relevant_doc_ids=["d1"],
            elapsed_sec=1.0,
        ),
    ]
    s = evaluate_results("test", qr, k=3)
    out = save_results_csv("test", s, qr, results_dir=tmp_path)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "q1" in text


def test_print_summary_does_not_raise(capsys) -> None:
    """print_summary が例外なくコンソール出力すること."""
    qr = [
        QueryResult(
            question_id="q1",
            retrieved_doc_ids=["d1"],
            relevant_doc_ids=["d1"],
            elapsed_sec=1.0,
        ),
    ]
    s = evaluate_results("test", qr, k=3)
    print_summary(s)
    captured = capsys.readouterr()
    assert "test" in captured.out
