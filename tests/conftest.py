"""pytest 共通フィクスチャ.

全テストはモック使用：sentence-transformers / chromadb / Ollama / cross-encoder の
ネットワーク・モデルロードなしで動作する。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# プロジェクトルートを sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# サンプル ドキュメント・質問
# ============================================================
@pytest.fixture
def sample_documents() -> list[dict]:
    """テスト用ナレッジベース（5件）."""
    return [
        {
            "id": "doc_001",
            "title": "ベアリング異音対応",
            "content": "スピンドルベアリングから異音が発生した場合、暖機運転とFFT分析で原因を切り分けます。",
            "category": "bearing_noise",
            "machine_type": "マシニングセンタ",
        },
        {
            "id": "doc_002",
            "title": "熱変位による寸法ズレ",
            "content": "連続加工で寸法がドリフトする現象は熱変位が原因です。暖機運転30分が基本対策です。",
            "category": "thermal_displacement",
            "machine_type": "マシニングセンタ",
        },
        {
            "id": "doc_003",
            "title": "原点復帰しない",
            "content": "原点復帰失敗時は原点センサ・エンコーダ・バッテリ・サーボの順で点検します。",
            "category": "motion_alarm",
            "machine_type": "マシニングセンタ",
        },
        {
            "id": "doc_004",
            "title": "錆発生防止",
            "content": "鉄部品の錆は防錆油塗布と低湿度保管で予防します。",
            "category": "surface_rust",
            "machine_type": "マシニングセンタ",
        },
        {
            "id": "doc_005",
            "title": "アンダー寸法対応",
            "content": "下限公差外れは廃却が原則です。切削条件・補正値・プログラムを点検します。",
            "category": "dimension_undersize",
            "machine_type": "マシニングセンタ",
        },
    ]


@pytest.fixture
def sample_questions() -> list[dict]:
    """テスト用質問（3件）."""
    return [
        {
            "id": "q_001",
            "question": "ベアリングから異音がする時の対応",
            "relevant_doc_ids": ["doc_001"],
            "answer": "FFT分析で切り分け",
        },
        {
            "id": "q_002",
            "question": "原点復帰しない時の対処",
            "relevant_doc_ids": ["doc_003"],
            "answer": "順次点検",
        },
        {
            "id": "q_003",
            "question": "錆を防ぐには",
            "relevant_doc_ids": ["doc_004"],
            "answer": "防錆油と低湿度",
        },
    ]


# ============================================================
# モック Embedder / LLMClient / Reranker
# ============================================================
class FakeEmbedder:
    """テスト用 Embedder.

    各テキストの先頭文字のコードポイントを使った決定的な低次元ベクトルを返す。
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """先頭文字のコードポイントベースで4次元ベクトル生成.

        Args:
            texts: 入力テキスト

        Returns:
            4次元ベクトル list
        """
        out: list[list[float]] = []
        for t in texts:
            base = ord(t[0]) if t else 0
            out.append([
                float(base % 100) / 100.0,
                float(base % 50) / 50.0,
                float(base % 25) / 25.0,
                float(base % 10) / 10.0,
            ])
        return out


class FakeLLMClient:
    """テスト用 LLMClient（決定的な応答を返す）."""

    def __init__(self, response: str = "モック応答") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


class FakeReranker:
    """テスト用 Reranker（先頭から降順スコアを返す）."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        # 先頭ほど高スコア（順位を逆転させない単純実装）
        n = len(candidates)
        return [float(n - i) for i in range(n)]


class FakeBM25:
    """テスト用 BM25 エンジン."""

    def __init__(self, n_corpus: int) -> None:
        self.n_corpus = n_corpus

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        # 単純に先頭から top_k 件を高スコアで返す
        return [(i, float(self.n_corpus - i)) for i in range(min(top_k, self.n_corpus))]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
