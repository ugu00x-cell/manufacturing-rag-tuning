"""baseline.py のテスト（モック使用）."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.baseline import (
    OllamaClient,
    SBertEmbedder,
    SYSTEM_PROMPT,
    build_index,
    build_user_prompt,
    chunk_documents,
    chunk_id_to_doc_id,
    chunk_text,
    retrieve,
)


# ============================================================
# チャンク分割
# ============================================================
class TestChunkText:
    """chunk_text のテスト."""

    def test_short_text_returns_single_chunk(self) -> None:
        """テキストがチャンクサイズ以下なら1チャンクで返る."""
        assert chunk_text("hello", chunk_size=100) == ["hello"]

    def test_chunks_with_overlap(self) -> None:
        """オーバーラップ付き分割が想定通り動くこと."""
        text = "abcdefghij"
        # size=4, overlap=2 → "abcd", "cdef", "efgh", "ghij", "ij"
        chunks = chunk_text(text, chunk_size=4, overlap=2)
        assert chunks[0] == "abcd"
        assert chunks[1] == "cdef"

    def test_invalid_chunk_size_raises(self) -> None:
        """chunk_size <= 0 で ValueError."""
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=0)

    def test_invalid_overlap_raises(self) -> None:
        """overlap >= chunk_size で ValueError."""
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_documents_preserves_doc_id(sample_documents: list[dict]) -> None:
    """全 doc_id がチャンク化結果に含まれること."""
    chunks = chunk_documents(sample_documents, chunk_size=100, overlap=0)
    doc_ids_in = {d["id"] for d in sample_documents}
    doc_ids_out = {doc_id for doc_id, _ in chunks}
    assert doc_ids_in == doc_ids_out


def test_chunk_id_to_doc_id_extracts_correctly() -> None:
    """chunk_id から doc_id が抽出できること."""
    assert chunk_id_to_doc_id("doc_001__chunk000") == "doc_001"
    assert chunk_id_to_doc_id("doc_999__chunk123") == "doc_999"


# ============================================================
# build_index + retrieve（FakeEmbedder + 実 ChromaDB）
# ============================================================
def test_build_index_and_retrieve(
    sample_documents: list[dict],
    fake_embedder,
) -> None:
    """ChromaDB + FakeEmbedder で検索が動くこと（実行時間短）."""
    index = build_index(sample_documents, embedder=fake_embedder, collection_name="test_baseline")
    retrieved = retrieve("ベアリング異音", index, fake_embedder, top_k=3)
    assert len(retrieved) <= 3
    # 全て (doc_id, chunk_text) のタプル
    for doc_id, chunk in retrieved:
        assert doc_id.startswith("doc_")
        assert isinstance(chunk, str)


# ============================================================
# プロンプト組み立て
# ============================================================
def test_build_user_prompt_includes_question_and_refs() -> None:
    """ユーザープロンプトに質問・参考文書が含まれること."""
    prompt = build_user_prompt(
        "テスト質問",
        retrieved=[("doc_001", "chunk1"), ("doc_002", "chunk2")],
    )
    assert "テスト質問" in prompt
    assert "chunk1" in prompt
    assert "chunk2" in prompt


def test_system_prompt_is_japanese() -> None:
    """システムプロンプトが日本語であること."""
    assert "日本語" in SYSTEM_PROMPT or "製造業" in SYSTEM_PROMPT


# ============================================================
# OllamaClient（HTTP リクエストをモック）
# ============================================================
class TestOllamaClient:
    """OllamaClient."""

    @patch("requests.post")
    def test_complete_returns_response(self, mock_post: MagicMock) -> None:
        """正常応答時に response 文字列が返ること."""
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"response": "テスト応答"}),
        )
        client = OllamaClient(model="gemma3:4b", base_url="http://localhost:11434", timeout=10)
        result = client.complete("system prompt", "user prompt")
        assert result == "テスト応答"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_complete_raises_on_connection_error(self, mock_post: MagicMock) -> None:
        """接続失敗時に RuntimeError."""
        import requests
        mock_post.side_effect = requests.ConnectionError("接続失敗")
        client = OllamaClient(timeout=5)
        with pytest.raises(RuntimeError, match="Ollama API 失敗"):
            client.complete("sys", "user")


# ============================================================
# SBertEmbedder（モデル読込はモックで回避）
# ============================================================
def test_sbert_embedder_lazy_loads() -> None:
    """初期化時点ではモデルロードが走らないこと."""
    e = SBertEmbedder()
    assert e._model is None
