"""ベースライン RAG（比較基準）.

仕様:
    - 埋め込み : sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
    - ベクトルDB : chromadb (永続化なし・in-memory)
    - チャンク : 固定500文字・オーバーラップなし
    - 検索 : コサイン類似度 Top3
    - 生成 : Ollama (gemma3:4b) — ローカルLLM・社外秘データOK
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)

# ============================================================
# 定数
# ============================================================
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 120
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_TOP_K = 3
DEFAULT_COLLECTION_NAME = "manufacturing_rag"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ============================================================
# 型定義
# ============================================================
class Document(TypedDict):
    """ナレッジベース文書."""

    id: str
    title: str
    content: str
    category: str
    machine_type: str


class Question(TypedDict):
    """評価用質問."""

    id: str
    question: str
    relevant_doc_ids: list[str]
    answer: str


class RAGResult(TypedDict):
    """1質問のRAG結果."""

    question_id: str
    retrieved_doc_ids: list[str]
    answer: str
    elapsed_sec: float


# ============================================================
# 抽象 Embedder（テスト時にモック可能）
# ============================================================
class Embedder(Protocol):
    """埋め込みベクトル化器プロトコル."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """文字列リストをベクトル化する."""
        ...


class SBertEmbedder:
    """sentence-transformers ベースの Embedder（実機用）."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """初期化（モデル読み込みは初回 embed 時に遅延）.

        Args:
            model_name: HuggingFace モデル名
        """
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        """モデルを遅延ロード."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Embedding model 読み込み: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """テキストをベクトル化.

        Args:
            texts: 入力テキスト

        Returns:
            ベクトル list
        """
        self._load()
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]


# ============================================================
# 抽象 LLMClient
# ============================================================
class LLMClient(Protocol):
    """LLM 応答生成プロトコル."""

    def complete(self, system: str, user: str) -> str:
        """システム/ユーザープロンプトから応答を生成する."""
        ...


class OllamaClient:
    """Ollama ローカルLLM ベースの LLMClient（実機用）.

    社外秘データを外部送信せずに RAG を構築するため、本プロジェクトでは
    Ollama + gemma3:4b をデフォルトの応答生成バックエンドとして採用する。

    必要条件:
        - Ollama がインストール・起動済み（http://localhost:11434）
        - 指定モデル（既定: gemma3:4b）が ollama pull 済み
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.1,
    ) -> None:
        """初期化.

        Args:
            model: Ollama モデル名（既定: gemma3:4b、env OLLAMA_MODEL でも上書き可）
            base_url: Ollama サーバー URL（既定: http://localhost:11434）
            timeout: HTTP タイムアウト秒数
            temperature: 生成温度（既定: 0.1）
        """
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).rstrip("/")
        env_timeout = os.environ.get("OLLAMA_TIMEOUT_SEC")
        self.timeout = timeout if timeout is not None else (
            int(env_timeout) if env_timeout else DEFAULT_OLLAMA_TIMEOUT
        )
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Ollama API で応答を生成する.

        Args:
            system: システムプロンプト
            user: ユーザープロンプト

        Returns:
            応答テキスト

        Raises:
            RuntimeError: Ollama サーバ接続失敗・モデル未取得など
        """
        import requests

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama API 失敗: {e} (Ollama起動と `ollama pull {self.model}` を確認)"
            ) from e

        data = r.json()
        return str(data.get("response", "")).strip()


# ============================================================
# チャンク分割
# ============================================================
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """文字数ベースでテキストをチャンクに分割する.

    Args:
        text: 入力テキスト
        chunk_size: 1チャンクの最大文字数
        overlap: チャンク間の重なり文字数

    Returns:
        チャンク文字列リスト
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size は正の整数である必要があります")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap は 0 以上 chunk_size 未満である必要があります")

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    pos = 0
    while pos < len(text):
        chunks.append(text[pos:pos + chunk_size])
        pos += step
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """全文書をチャンク化する.

    Args:
        documents: ナレッジベース文書
        chunk_size: チャンクサイズ
        overlap: オーバーラップ

    Returns:
        [(doc_id, chunk_text), ...]（同一doc_idが複数チャンクで現れる）
    """
    out: list[tuple[str, str]] = []
    for doc in documents:
        # title を文頭に付与して検索精度を上げる
        full = f"{doc['title']}\n{doc['content']}"
        for chunk in chunk_text(full, chunk_size=chunk_size, overlap=overlap):
            out.append((doc["id"], chunk))
    return out


# ============================================================
# ChromaDB ラッパ
# ============================================================
class ChromaIndex:
    """ChromaDB を使ったベクトル検索インデックス."""

    def __init__(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        """初期化.

        Args:
            collection_name: コレクション名
        """
        import chromadb
        # in-memory（永続化なし）
        self.client = chromadb.EphemeralClient()
        # 既存コレクション削除して新規作成
        try:
            self.client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 — chromadb の例外型は版で変わる
            pass
        # cosine 類似度を明示
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]]) -> None:
        """ドキュメントを登録する.

        Args:
            ids: ユニークID（チャンク単位）
            texts: チャンクテキスト
            embeddings: 対応する埋め込みベクトル
        """
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)

    def query(self, embedding: list[float], top_k: int = DEFAULT_TOP_K) -> list[tuple[str, str, float]]:
        """ベクトル検索を実行.

        Args:
            embedding: 質問の埋め込みベクトル
            top_k: 取得件数

        Returns:
            [(chunk_id, chunk_text, distance), ...]
        """
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        dists = result["distances"][0] if result.get("distances") else [0.0] * len(ids)
        return list(zip(ids, docs, dists))


# ============================================================
# パイプライン
# ============================================================
def build_index(
    documents: list[Document],
    embedder: Embedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> ChromaIndex:
    """ナレッジベースから ChromaIndex を構築する.

    Args:
        documents: ナレッジベース文書
        embedder: 埋め込みベクトル化器
        chunk_size: チャンクサイズ
        overlap: オーバーラップ
        collection_name: コレクション名

    Returns:
        ChromaIndex
    """
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    chunk_ids = [f"{doc_id}__chunk{i:03d}" for i, (doc_id, _) in enumerate(chunks)]
    chunk_texts = [t for _, t in chunks]

    logger.info("Embedding %d チャンクを生成中...", len(chunk_texts))
    embeddings = embedder.embed(chunk_texts)

    index = ChromaIndex(collection_name=collection_name)
    index.add(chunk_ids, chunk_texts, embeddings)
    logger.info("Index 構築完了: %d チャンク", len(chunk_ids))
    return index


def chunk_id_to_doc_id(chunk_id: str) -> str:
    """chunk_id から doc_id を抽出する（"doc_001__chunk000" → "doc_001"）.

    Args:
        chunk_id: チャンクID

    Returns:
        ドキュメントID
    """
    return chunk_id.split("__")[0]


def retrieve(
    question: str,
    index: ChromaIndex,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, str]]:
    """質問に対して検索を実行する.

    Args:
        question: 自然言語の質問
        index: 検索インデックス
        embedder: Embedder
        top_k: 取得件数（doc単位での重複除去後）

    Returns:
        [(doc_id, chunk_text), ...]（重複除去後・上位 top_k 件）
    """
    # 重複除去のため多めに取得
    raw = index.query(embedder.embed([question])[0], top_k=top_k * 3)

    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for chunk_id, chunk_text_, _dist in raw:
        doc_id = chunk_id_to_doc_id(chunk_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append((doc_id, chunk_text_))
        if len(result) >= top_k:
            break
    return result


SYSTEM_PROMPT = """あなたは製造業の不良対応に詳しいエンジニアです。
与えられた参考文書のみを根拠に、質問に対して簡潔に日本語で回答してください。
参考文書に記載がない情報は推測せず「資料に記載なし」と答えてください。"""


def build_user_prompt(question: str, retrieved: list[tuple[str, str]]) -> str:
    """ユーザープロンプトを組み立てる.

    Args:
        question: 質問
        retrieved: 検索結果

    Returns:
        ユーザープロンプト
    """
    refs = "\n\n".join(f"[文書{i + 1}] {chunk}" for i, (_, chunk) in enumerate(retrieved))
    return f"参考文書:\n{refs}\n\n質問: {question}\n\n回答:"


def answer_question(
    question: Question,
    index: ChromaIndex,
    embedder: Embedder,
    llm: LLMClient,
    top_k: int = DEFAULT_TOP_K,
) -> RAGResult:
    """1質問に対して RAG パイプラインを実行する.

    Args:
        question: 評価用質問
        index: 検索インデックス
        embedder: Embedder
        llm: LLMClient
        top_k: 検索件数

    Returns:
        RAGResult
    """
    start = time.perf_counter()
    retrieved = retrieve(question["question"], index, embedder, top_k=top_k)
    answer_text = llm.complete(SYSTEM_PROMPT, build_user_prompt(question["question"], retrieved))
    elapsed = time.perf_counter() - start

    return RAGResult(
        question_id=question["id"],
        retrieved_doc_ids=[doc_id for doc_id, _ in retrieved],
        answer=answer_text,
        elapsed_sec=round(elapsed, 3),
    )


def load_data(
    data_dir: Path = DATA_DIR,
) -> tuple[list[Document], list[Question]]:
    """JSON ファイルから documents / questions をロードする.

    Args:
        data_dir: データディレクトリ

    Returns:
        (documents, questions)
    """
    docs_path = data_dir / "documents.json"
    qs_path = data_dir / "questions.json"
    if not docs_path.exists() or not qs_path.exists():
        raise FileNotFoundError(
            f"データが見つかりません。先に `python -m src.generate_data` を実行してください: {docs_path}"
        )
    documents = json.loads(docs_path.read_text(encoding="utf-8"))
    questions = json.loads(qs_path.read_text(encoding="utf-8"))
    return documents, questions


def run_baseline_evaluation(
    documents: list[Document],
    questions: list[Question],
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    top_k: int = DEFAULT_TOP_K,
    save_csv: bool = True,
    skip_generation: bool = False,
) -> tuple[dict, list]:
    """全質問に対してベースライン RAG を評価する.

    Args:
        documents: ナレッジベース
        questions: 評価質問
        embedder: Embedder（None なら SBertEmbedder）
        llm: LLMClient（None なら OllamaClient）
        top_k: 検索件数
        save_csv: 結果を CSV 保存するか
        skip_generation: True なら検索のみで生成は省略（API/Ollama不要）

    Returns:
        (サマリー, 各質問の QueryResult)
    """
    from src.evaluate import (
        QueryResult,
        evaluate_results,
        print_summary,
        save_results_csv,
    )

    embedder = embedder or SBertEmbedder()
    index = build_index(documents, embedder=embedder, collection_name="baseline_idx")

    # 生成するなら LLM を初期化
    if not skip_generation and llm is None:
        llm = OllamaClient()

    query_results: list[QueryResult] = []
    for q in questions:
        start = time.perf_counter()
        retrieved = retrieve(q["question"], index, embedder, top_k=top_k)
        if not skip_generation and llm is not None:
            _ = llm.complete(SYSTEM_PROMPT, build_user_prompt(q["question"], retrieved))
        elapsed = time.perf_counter() - start

        query_results.append(QueryResult(
            question_id=q["id"],
            retrieved_doc_ids=[d for d, _ in retrieved],
            relevant_doc_ids=q["relevant_doc_ids"],
            elapsed_sec=round(elapsed, 4),
        ))

    summary = evaluate_results(method="baseline", query_results=query_results, k=top_k)
    print_summary(summary)
    if save_csv:
        save_results_csv("baseline", summary, query_results)
    return dict(summary), query_results


def main() -> None:
    """エントリポイント.

    使い方:
        python -m src.baseline                # 検索のみ評価（Ollama不要）
        python -m src.baseline --with-generation  # 生成込み（Ollama必須）
    """
    import argparse
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser(description="ベースライン RAG 評価")
    p.add_argument(
        "--with-generation",
        action="store_true",
        help="Ollama で生成も実行（要 ollama serve + gemma3:4b）",
    )
    args = p.parse_args()

    documents, questions = load_data()
    run_baseline_evaluation(
        documents, questions, skip_generation=not args.with_generation
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
