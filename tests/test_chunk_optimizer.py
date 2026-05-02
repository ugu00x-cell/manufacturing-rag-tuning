"""chunk_optimizer.py のテスト."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunk_optimizer import PATTERNS, evaluate_pattern, run_all_patterns


# ============================================================
# 正常系
# ============================================================
class TestNormal:
    """正常系."""

    def test_patterns_count(self) -> None:
        """仕様通り5パターン定義されていること."""
        assert len(PATTERNS) == 5

    def test_pattern_names_unique(self) -> None:
        """パターン名が重複していないこと."""
        names = [p["name"] for p in PATTERNS]
        assert len(set(names)) == len(names)

    def test_evaluate_single_pattern(
        self,
        sample_documents: list[dict],
        sample_questions: list[dict],
        fake_embedder,
    ) -> None:
        """単一パターンが評価結果を返すこと."""
        pattern = PATTERNS[0]
        summary, results = evaluate_pattern(
            pattern, sample_documents, sample_questions, fake_embedder, top_k=3
        )
        assert summary["method"] == f"chunk_{pattern['name']}"
        assert summary["n_questions"] == len(sample_questions)
        assert len(results) == len(sample_questions)


# ============================================================
# 異常系
# ============================================================
class TestAbnormal:
    """異常系."""

    def test_empty_questions_returns_zero_summary(
        self,
        sample_documents: list[dict],
        fake_embedder,
    ) -> None:
        """質問が空でも処理できること."""
        summary, results = evaluate_pattern(
            PATTERNS[0], sample_documents, [], fake_embedder, top_k=3
        )
        assert summary["n_questions"] == 0
        assert results == []


# ============================================================
# 境界値
# ============================================================
class TestBoundary:
    """境界値."""

    def test_smallest_chunk_pattern(self) -> None:
        """最小チャンクサイズが200であること."""
        smallest = min(p["chunk_size"] for p in PATTERNS)
        assert smallest == 200

    def test_largest_chunk_pattern(self) -> None:
        """最大チャンクサイズが1000であること."""
        largest = max(p["chunk_size"] for p in PATTERNS)
        assert largest == 1000


# ============================================================
# 全パターン実行 + CSV保存
# ============================================================
def test_run_all_patterns_writes_csv(
    sample_documents: list[dict],
    sample_questions: list[dict],
    fake_embedder,
    tmp_path: Path,
) -> None:
    """全パターン実行で CSV が出力されること."""
    out_path = run_all_patterns(
        sample_documents, sample_questions, embedder=fake_embedder, results_dir=tmp_path
    )
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    # 各パターンが行として含まれる
    for p in PATTERNS:
        assert p["name"] in text
