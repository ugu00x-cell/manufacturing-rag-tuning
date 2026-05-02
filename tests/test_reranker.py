"""reranker.py のテスト（FakeReranker使用）."""

from __future__ import annotations

import pytest

from src.baseline import build_index
from src.reranker import (
    CrossEncoderReranker,
    retrieve_with_rerank,
    run_rerank_evaluation,
)


# ============================================================
# 正常系
# ============================================================
class TestNormal:
    """正常系."""

    def test_retrieve_with_rerank_returns_final_top_k(
        self,
        sample_documents: list[dict],
        fake_embedder,
        fake_reranker,
    ) -> None:
        """最終 Top-K が返ること."""
        index = build_index(sample_documents, embedder=fake_embedder, collection_name="rerank_test")
        result = retrieve_with_rerank(
            "ベアリング異音",
            index=index,
            embedder=fake_embedder,
            reranker=fake_reranker,
            initial_top_k=10,
            final_top_k=2,
        )
        assert len(result) <= 2

    def test_run_rerank_evaluation_returns_summary(
        self,
        sample_documents: list[dict],
        sample_questions: list[dict],
        fake_embedder,
        fake_reranker,
    ) -> None:
        """評価サマリーが返ること."""
        summary, results = run_rerank_evaluation(
            sample_documents,
            sample_questions,
            embedder=fake_embedder,
            reranker=fake_reranker,
            save_csv=False,
        )
        assert summary["method"] == "reranker"
        assert summary["n_questions"] == len(sample_questions)
        assert len(results) == len(sample_questions)


# ============================================================
# 異常系
# ============================================================
class TestAbnormal:
    """異常系."""

    def test_empty_candidates_handled(self, fake_reranker) -> None:
        """候補が空でも reranker が壊れないこと."""
        scores = fake_reranker.rerank("query", [])
        assert scores == []


# ============================================================
# 境界値
# ============================================================
class TestBoundary:
    """境界値."""

    def test_cross_encoder_lazy_load(self) -> None:
        """初期化時点でモデルロードが走らないこと."""
        r = CrossEncoderReranker()
        assert r._model is None

    def test_single_candidate_returns_single_score(self, fake_reranker) -> None:
        """候補1件なら1スコア返る."""
        scores = fake_reranker.rerank("query", ["text1"])
        assert len(scores) == 1
