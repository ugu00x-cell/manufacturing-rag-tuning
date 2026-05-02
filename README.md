# manufacturing-rag-tuning

> 製造業の不良対応データを題材に、**RAGの精度改善手法**をベースラインから段階的に実装・比較するリポジトリです。
>
> 「**RAGを作ったけど回答がズレる**」という現場課題に対する、**実装ベースの解決策**を提示します。

---

## 🔒 社外APIなし・ローカルLLMで完結

製造業の不良対応データには、**型番・顧客名・社内コード**が混入しがちで、
情報システム部門が外部API送信を許可しないケースが大半です。

本リポジトリは **Ollama + gemma3:4b** をデフォルトの応答生成バックエンドに採用し、
**ベクトルDB (ChromaDB)** と **多言語埋め込み (sentence-transformers)** を組み合わせて、
**社外秘データを外部送信せずに RAG を構築**できることを示します。

> 関連リポジトリ：[defect-text-classification](https://github.com/ugu00x-cell/defect-text-classification) — 不良報告書の自動分類（Claude × Ollama 比較）

---

## 背景：「RAGが当たらない」現場課題

社内ドキュメントRAGを構築したものの、以下のような問題が頻発しがちです：

- 質問と関係のない文書が Top-K に紛れ込む
- 質問の **言い換え**（「異音」⇄「ガタガタ音」）に弱い
- チャンクサイズが大きすぎて関係ない情報まで含まれる/小さすぎて文脈が断片化する

本リポジトリでは、これらに対する以下4手法を実装・比較します：

| # | 手法 | 内容 | 期待効果 |
|---|------|------|---------|
| 1 | **baseline** | 固定500文字チャンク + ベクトル検索 Top-3 | 基準値（比較ベース） |
| 2 | **chunk_optimizer** | 5パターンのチャンク設定で精度比較 | 最適チャンクサイズ判明 |
| 3 | **reranker** | ベクトル検索 Top-10 → CrossEncoder で並べ替え → Top-3 | 関連性の高い文書が上位に |
| 4 | **hybrid_search** | ベクトル検索 + BM25キーワード検索 → RRF統合 | 言い換え/語彙ミスマッチに強い |

---

## 📐 技術構成

```
manufacturing-rag-tuning/
├── data/
│   ├── documents.json       # ナレッジベース 50件 (8カテゴリ × 3機械タイプ)
│   └── questions.json       # 評価Q&Aペア 20件 (正解 doc_ids 付き)
├── src/
│   ├── generate_data.py     # 合成データ生成
│   ├── baseline.py          # ベースライン RAG（Ollama）
│   ├── chunk_optimizer.py   # チャンクサイズ最適化（5パターン比較）
│   ├── reranker.py          # CrossEncoder リランキング
│   ├── hybrid_search.py     # ベクトル + BM25 ハイブリッド (RRF)
│   └── evaluate.py          # 共通評価モジュール (Recall@K / MRR / 応答時間)
├── tests/                   # pytest 44テスト全モック使用
├── notebooks/
│   └── comparison.ipynb     # 4手法比較の可視化
├── results/                 # 評価結果CSV (gitignore対象)
├── requirements.txt
├── .env.example
└── README.md
```

### 採用技術

| レイヤ | 技術 | 理由 |
|--------|------|------|
| 埋め込み | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 日本語含む多言語対応・OSS |
| ベクトルDB | ChromaDB (in-memory) | セットアップ不要・cosine明示指定 |
| キーワード検索 | rank_bm25 + 自前トークナイザ | 形態素解析器導入なしで動作 |
| リランカー | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 軽量・CPU動作可 |
| 応答生成 | **Ollama + gemma3:4b** | ローカル完結・社外秘OK・APIコスト0 |

---

## ⚙️ セットアップ

### 1. リポジトリ取得

```bash
git clone https://github.com/ugu00x-cell/manufacturing-rag-tuning.git
cd manufacturing-rag-tuning
```

### 2. 依存インストール

```bash
# uv 推奨
uv venv
.\.venv\Scripts\activate          # Windows PowerShell
uv pip install -r requirements.txt

# pip でも可
# pip install -r requirements.txt
```

⚠️ **初回 pip install は 10〜20 分**かかります（torch + transformers + chromadb 計 3〜4GB）。

### 3. Ollama インストール & モデル取得

```bash
# Windows
winget install Ollama.Ollama
# macOS
# brew install ollama
# Linux
# curl -fsSL https://ollama.com/install.sh | sh

# モデル取得（約 3GB）
ollama pull gemma3:4b

# サーバ起動（別ターミナル）
ollama serve
```

### 4. モデルキャッシュの場所（任意）

`sentence-transformers` と `cross-encoder` のモデルは初回実行時に自動DLされます（合計 **約1.2GB**）。
キャッシュ先を変えたい場合は `.env` で：

```env
SENTENCE_TRANSFORMERS_HOME=D:/models/sentence-transformers
HF_HOME=D:/models/huggingface
```

### 5. 環境変数

```bash
cp .env.example .env
# .env を開いて Ollama URL 等を確認（既定値で OK）
```

---

## 🚀 実行手順

```bash
# 0. 合成データ生成
python -m src.generate_data
# → data/documents.json (50件) / data/questions.json (20件)

# 1. ベースライン評価（検索のみ・Ollama不要）
python -m src.baseline

# 1'. ベースライン評価（生成込み・要 Ollama）
python -m src.baseline --with-generation

# 2. チャンク5パターン比較
python -m src.chunk_optimizer
# → results/chunk_optimizer_summary.csv

# 3. リランキング評価
python -m src.reranker
# → results/reranker_<timestamp>.csv

# 4. ハイブリッド検索評価
python -m src.hybrid_search
# → results/hybrid_search_<timestamp>.csv

# 5. 可視化ノートブック
jupyter notebook notebooks/comparison.ipynb
```

---

## 📊 比較結果サマリ（プレースホルダ）

実行後にこのテーブルを更新してください：

| 手法 | Recall@3 | MRR | 平均応答時間 |
|------|---------|-----|------------|
| baseline | （実行後追記） | （実行後追記） | （実行後追記） |
| chunk_500_overlap100 | （実行後追記） | （実行後追記） | （実行後追記） |
| reranker | （実行後追記） | （実行後追記） | （実行後追記） |
| hybrid_search | （実行後追記） | （実行後追記） | （実行後追記） |

> 一般的傾向（参考値）:
> - reranker は baseline より Recall@3 が **+10〜20pt** 改善することが多い
> - hybrid_search は語彙の揺れに強く、専門用語クエリで効果大
> - チャンク 500文字 + オーバーラップ 100文字 が日本語では実用ベンチでの安定解

---

## 🧪 テスト

```bash
pytest                       # 全テスト実行
pytest -v                    # 詳細
pytest --cov=src --cov-report=html
```

**44テストPASS**。実機モデル・Ollama 起動なしで動作する設計です（FakeEmbedder / FakeReranker / requests.post モック）。

---

## 🎯 各手法の使い分けガイド

| シーン | 推奨手法 | 理由 |
|--------|---------|------|
| **すぐに使える MVP が欲しい** | `baseline` | 最少コードで構築可能 |
| **チャンクサイズが分からない** | `chunk_optimizer` | 自社データに最適なサイズが判明 |
| **質問と関連の薄い文書が混じる** | `reranker` | 関連度を二段階で評価 |
| **専門用語の言い換えに弱い** | `hybrid_search` | キーワード検索で補完 |
| **本番運用で精度最大化したい** | `reranker + hybrid_search` の組合せ | 各層で異なる弱点を補完（Phase 2 候補） |

---

## 🛣 今後の展望

### Phase 2 候補

- [ ] reranker × hybrid_search の組合せ（Phase 1.5）
- [ ] ColBERT / 多段階リランキングの比較
- [ ] [defect-text-classification](https://github.com/ugu00x-cell/defect-text-classification) のデータ統合
  - 不良分類結果（カテゴリ・小分類）を**フィルタとして使う**ハイブリッド検索（事前カテゴリ絞込み → ハイブリッド検索）
  - 想定改善：Recall@3 が 5〜10pt 向上見込み
- [ ] HyDE (Hypothetical Document Embeddings) 実装
- [ ] 質問展開（クエリ拡張）の効果検証

### Phase 3 候補（実用化）

- [ ] Streamlit UI で現場デモ
- [ ] kintone / Notion / Confluence 等のドキュメント取り込みコネクタ
- [ ] 評価データの拡充（実データ匿名化スキーマ）

---

## 📂 関連リポジトリ

| 系統 | リポジトリ | 対象 |
|------|----------|------|
| 数値データ系 | [manufacturing-ai-toolkit](https://github.com/ugu00x-cell/manufacturing-ai-toolkit) | 振動・音響・加工条件 |
| 振動異常検知 | [bearing-anomaly-detection](https://github.com/ugu00x-cell/bearing-anomaly-detection) | CWRU / NASA データセット |
| 自然言語分類 | [defect-text-classification](https://github.com/ugu00x-cell/defect-text-classification) | 不良報告書の Claude × Ollama 比較 |
| **RAG 精度改善（本リポ）** | manufacturing-rag-tuning | 4手法比較 |
| 予知保全API | [predictive-maintenance-api](https://github.com/ugu00x-cell/predictive-maintenance-api) | FastAPI 基盤 |

---

## 📜 ライセンス

MIT License

---

## 著者

**竹中純也 (Junya Takenaka)**
- 製造業15年（加工技術5年 + 品質管理10年）
- GitHub: [@ugu00x-cell](https://github.com/ugu00x-cell)
- Zenn: [zenn.dev/ugu000x](https://zenn.dev/ugu000x)
