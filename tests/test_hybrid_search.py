"""hybrid_search.py のテスト."""

from __future__ import annotations

import pytest

from src.baseline import build_index, chunk_documents
from src.hybrid_search import (
    build_bm25_corpus,
    hybrid_retrieve,
    reciprocal_rank_fusion,
    run_hybrid_evaluation,
    simple_tokenize,
)


# ============================================================
# トークナイザ
# ============================================================
class TestTokenize:
    """simple_tokenize."""

    def test_japanese_only(self) -> None:
        """日本語が文字単位で分割されること."""
        tokens = simple_tokenize("ベアリング異音")
        assert "ベ" in tokens
        assert "異" in tokens

    def test_alphanumeric(self) -> None:
        """英数字がまとまったトークンになること."""
        tokens = simple_tokenize("ID123 abc")
        assert "id123" in tokens
        assert "abc" in tokens

    def test_empty_text(self) -> None:
        """空文字でもエラーなく動作すること."""
        assert simple_tokenize("") == []


# ============================================================
# RRF
# ============================================================
class TestRRF:
    """reciprocal_rank_fusion."""

    def test_single_ranking_returns_descending_score(self) -> None:
        """単一ランキングでも順位通りスコアが降順."""
        result = reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [d for d, _ in result]
        assert ids == ["a", "b", "c"]

    def test_two_rankings_combine_scores(self) -> None:
        """両ランキングで上位の文書が最高スコア."""
        result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
        # "a" は両方で1位なので合計スコアが最大
        assert result[0][0] == "a"

    def test_disjoint_rankings(self) -> None:
        """重複のないランキングを統合."""
        result = reciprocal_rank_fusion([["a"], ["b"]])
        ids = [d for d, _ in result]
        assert set(ids) == {"a", "b"}


# ============================================================
# BM25 構築 + ハイブリッド検索
# ============================================================
def test_build_bm25_corpus(sample_documents: list[dict]) -> None:
    """BM25 コーパス構築が成功すること."""
    bm25, chunk_ids = build_bm25_corpus(sample_documents)
    assert bm25 is not None
    assert len(chunk_ids) > 0


def test_hybrid_retrieve_returns_doc_ids(
    sample_documents: list[dict],
    fake_embedder,
) -> None:
    """ハイブリッド検索が doc_id list を返すこと."""
    index = build_index(sample_documents, embedder=fake_embedder, collection_name="hybrid_test")
    bm25, chunk_ids = build_bm25_corpus(sample_documents)
    result = hybrid_retrieve(
        "ベアリング異音",
        index=index,
        embedder=fake_embedder,
        bm25=bm25,
        chunk_ids=chunk_ids,
    )
    assert isinstance(result, list)
    assert all(d.startswith("doc_") for d in result)


def test_run_hybrid_evaluation(
    sample_documents: list[dict],
    sample_questions: list[dict],
    fake_embedder,
) -> None:
    """ハイブリッド検索の総合評価が動くこと."""
    summary, results = run_hybrid_evaluation(
        sample_documents,
        sample_questions,
        embedder=fake_embedder,
        save_csv=False,
    )
    assert summary["method"] == "hybrid_search"
    assert len(results) == len(sample_questions)


# ============================================================
# FakeBM25 を使った軽量テスト
# ============================================================
class TestWithFakeBM25:
    """FakeBM25 で hybrid_retrieve を動かす."""

    def test_hybrid_with_fake_bm25(
        self,
        sample_documents: list[dict],
        fake_embedder,
    ) -> None:
        """FakeBM25 でも例外なく動作すること."""
        from tests.conftest import FakeBM25

        index = build_index(sample_documents, embedder=fake_embedder, collection_name="hybrid_fake_bm25")
        chunks = chunk_documents(sample_documents)
        chunk_ids = [f"{doc_id}__chunk{i:03d}" for i, (doc_id, _) in enumerate(chunks)]
        fake_bm25 = FakeBM25(n_corpus=len(chunk_ids))
        result = hybrid_retrieve(
            "test",
            index=index,
            embedder=fake_embedder,
            bm25=fake_bm25,
            chunk_ids=chunk_ids,
        )
        assert isinstance(result, list)
