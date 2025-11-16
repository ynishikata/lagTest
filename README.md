## lagTest – OpenAI RAG API Prototype

### **概要**

このリポジトリは、OpenAI API を使って **RAG（Retrieval-Augmented Generation）** を行うシンプルな HTTP サーバーです。  
任意のテキスト文書を API 経由で投入し、その内容に基づいて質問に回答できます。

- **言語**: Go
- **依存サービス**: OpenAI API（埋め込み + Chat Completions）

---

### **提供機能**

- **ブラウザUI**
  - パス: `GET /`（`static/index.html`）
  - 機能:
    - 画面左で文書を貼り付けて「文書を登録する」ボタンから `/documents` にPOST
    - 画面右で質問文・`top_k`・任意の system prompt を入力して「質問する」ボタンから `/query` にPOST
    - 回答と、回答に利用されたコンテキストチャンク一覧をその場で確認可能

- **文書インジェスト（登録）API**
  - エンドポイント: `POST /documents`
  - リクエスト:
    - `content`: 登録したいテキスト全文（UTF-8）
  - 処理内容:
    - 文書を約 800 文字ごとにチャンク分割
    - 各チャンクに対して `text-embedding-3-small` で埋め込みベクトルを作成
    - 生成したチャンクとベクトルを **メモリ上の簡易ベクターストア** に保存
  - レスポンス:
    - `document_id`: 付与された文書ID（現在はあくまで識別用）

- **RAG クエリ（質問）API**
  - エンドポイント: `POST /query`
  - リクエスト:
    - `query`: 質問文
    - `top_k` (オプション): 類似チャンクの上位何件をコンテキストに使うか（デフォルト 3）
    - `system_prompt` (オプション): モデルに対するシステムプロンプト（未指定なら「コンテキストのみを根拠に答える」プロンプトを使用）
  - 処理内容:
    - `query` を埋め込みモデルに投げてベクトル化
    - コサイン類似度で、メモリ上に保存されているチャンクの中から上位 `top_k` 件を取得
    - それらのチャンクをまとめてコンテキストとして `gpt-4.1-mini` に投げ、回答を生成
  - レスポンス:
    - `answer`: モデルが生成した回答
    - `sources`: 回答に使われたコンテキストチャンクの配列（どの文脈から答えたかの簡易トレース用）

**注意**: すべてのデータはプロセス内のメモリにのみ保持されており、永続化はしていません。サーバーを再起動すると文書/埋め込みは失われます。

---

### **ソフトウェア構造**

メインの実装は `main.go` のみで完結しており、小さなプロジェクト構成になっています。

- **`document` / `storedChunk` / `ragStore`**
  - `document`
    - 元の文書全文と、その文書から切り出した `Chunks` の配列を持つ構造体。
  - `storedChunk`
    - 文書ID (`DocID`)、チャンクのテキスト (`Text`)、埋め込みベクトル (`Vector []float32`) を持つ。
  - `ragStore`
    - `docs`: インジェストされた `document` の配列
    - `chunks`: 検索対象の `storedChunk` の配列
    - `nextDoc`: 次に採番する文書ID
    - 内部ロックとして `sync.RWMutex` を使用し、並行アクセスを保護。

- **類似度計算**
  - `cosineSimilarity(a, b []float32) float32`
    - 2 つのベクトル間のコサイン類似度を計算。
  - `sqrt(x float32) float32`
    - ニュートン法による簡易平方根計算（外部依存を避けるための軽量実装）。

- **API リクエスト/レスポンス構造体**
  - `ingestRequest` / `ingestResponse`
    - `/documents` 用の JSON 形式リクエスト/レスポンス。
  - `queryRequest` / `queryResponse`
    - `/query` 用の JSON 形式リクエスト/レスポンス。

- **OpenAI API 呼び出しラッパ**
  - `createEmbedding(ctx, apiKey, text) ([]float32, error)`
    - `https://api.openai.com/v1/embeddings` に対して HTTP POST を行い、`text-embedding-3-small` で埋め込みベクトルを取得。
    - レスポンス JSON を `[]float32` のスライスとして返す。
  - `chatCompletion(ctx, apiKey, systemPrompt, userContent) (string, error)`
    - `https://api.openai.com/v1/chat/completions` に対して HTTP POST を行い、`gpt-4.1-mini` から回答テキストを取得。
    - `system` ロールに `systemPrompt`、`user` ロールに「Context + Question」を渡す。

- **HTTP ハンドラ**
  - `GET /health`
    - シンプルなヘルスチェック。`200 OK` と `ok` を返す。
  - `POST /documents`
    - JSON パース → チャンク分割 → 埋め込み生成 → `ragStore` への格納 → `document_id` を返す。
  - `POST /query`
    - JSON パース → 質問の埋め込み → 既存チャンクとの類似度スコア計算 → 上位 `top_k` のチャンクをコンテキストとして Chat Completion 呼び出し → 回答と使用チャンクを返す。

---

### **セットアップと起動方法**

1. **依存関係の取得**

```bash
go mod tidy
```

2. **OpenAI API キーの設定**

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

3. **サーバーの起動**

```bash
go run main.go
```

ログに `Server starting on :8080` が出ていれば成功です。

---

### **簡単な利用例**

- **文書インジェスト**

```bash
curl -X POST http://localhost:8080/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "これはテスト用の文書です。RAGの動作確認のためのテキストが入っています。"}'
```

- **質問クエリ**

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "このテスト文書は何について書かれていますか？",
    "top_k": 3
  }'
```

---

### **今後の拡張アイデア**

- 永続化ストレージ対応（SQLite や Postgres など）  
- 文書ごとのスコープ指定（特定 `document_id` のみを検索対象にする機能）  
- OpenAPI (Swagger) 定義の追加  
- 簡単な Web UI（ブラウザからファイルアップロード＋質問ができる画面）  


