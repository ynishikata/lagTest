package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// simple in-memory vector store
type document struct {
	ID      int      `json:"id"`
	Content string   `json:"content"`
	Chunks  []string `json:"chunks"`
}

type storedChunk struct {
	DocID  int
	Text   string
	Vector []float32
}

type ragStore struct {
	mu      sync.RWMutex
	chunks  []storedChunk
	docs    []document
	nextDoc int
}

// --- Meeting / multi-agent discussion types ---

// Message represents a single utterance in the virtual meeting.
type Message struct {
	Round   int    `json:"round"`
	Speaker string `json:"speaker"`
	Content string `json:"content"`
}

// AgentConfig defines basic profile for an agent.
type AgentConfig struct {
	Name        string
	RoleSummary string
}

// MeetingAgent is an interface for agents participating in the discussion.
type MeetingAgent interface {
	Name() string
	Speak(ctx context.Context, apiKey string, topic string, logs []Message) (string, error)
}

// gptAgent is a simple implementation backed by OpenAI Chat API.
type gptAgent struct {
	cfg AgentConfig
}

func (a *gptAgent) Name() string { return a.cfg.Name }

func (a *gptAgent) Speak(ctx context.Context, apiKey string, topic string, logs []Message) (string, error) {
	var buf strings.Builder
	for _, m := range logs {
		fmt.Fprintf(&buf, "Round %d [%s]: %s\n", m.Round, m.Speaker, m.Content)
	}
	historyText := buf.String()

	systemPrompt := fmt.Sprintf(
		`あなたは仮想会議に参加するAIです。
あなたの名前: %s
あなたの役割: %s

次のフォーマットで必ず日本語で発言してください:

[理由]
- 箇条書きで2〜4点

[提案]
- 1つの具体的な案

[質問]
- 他の参加者に投げる質問を1つ

同じことを繰り返さず、議論を前に進めてください。`,
		a.cfg.Name,
		a.cfg.RoleSummary,
	)

	userContent := fmt.Sprintf(
		"会議トピック:\n%s\n\nこれまでの議論ログ:\n%s\n\n上記を踏まえて、あなたの次の発言だけをフォーマットに従って返してください。",
		topic,
		historyText,
	)

	return chatCompletion(ctx, apiKey, systemPrompt, userContent)
}

// MeetingManager orchestrates a multi-agent discussion.
type MeetingManager struct {
	Agents    []MeetingAgent
	MaxRounds int
	MaxSilent int
}

// MeetingResult holds the final logs and summary.
type MeetingResult struct {
	Topic   string    `json:"topic"`
	Logs    []Message `json:"logs"`
	Summary string    `json:"summary"`
}

// RunMeeting runs the discussion loop.
func (m *MeetingManager) RunMeeting(ctx context.Context, apiKey string, topic string) (*MeetingResult, error) {
	var logs []Message
	round := 1
	silentRounds := 0

	// initial user topic log
	logs = append(logs, Message{
		Round:   0,
		Speaker: "User",
		Content: topic,
	})

	for round <= m.MaxRounds {
		roundHasNewInfo := false

		for _, agent := range m.Agents {
			content, err := agent.Speak(ctx, apiKey, topic, logs)
			if err != nil {
				return nil, err
			}
			if strings.TrimSpace(content) == "" {
				continue
			}

			logs = append(logs, Message{
				Round:   round,
				Speaker: agent.Name(),
				Content: content,
			})
			roundHasNewInfo = true
		}

		if roundHasNewInfo {
			silentRounds = 0
		} else {
			silentRounds++
			if silentRounds >= m.MaxSilent {
				break
			}
		}

		round++
	}

	summary, err := generateSummary(ctx, apiKey, topic, logs)
	if err != nil {
		return nil, err
	}

	return &MeetingResult{
		Topic:   topic,
		Logs:    logs,
		Summary: summary,
	}, nil
}

// generateSummary creates the final meeting summary.
func generateSummary(ctx context.Context, apiKey, topic string, logs []Message) (string, error) {
	var buf strings.Builder
	for _, m := range logs {
		fmt.Fprintf(&buf, "Round %d [%s]: %s\n", m.Round, m.Speaker, m.Content)
	}
	logText := buf.String()

	systemPrompt := `あなたは会議モデレーターです。
以下の議論ログを読み、「結論・利点・リスク・推奨案」の4項目で日本語のサマリーを作成してください。
それぞれ見出しを付けて、箇条書きを中心に端的にまとめてください。`

	userContent := fmt.Sprintf(
		"会議トピック:\n%s\n\n議論ログ:\n%s",
		topic,
		logText,
	)

	return chatCompletion(ctx, apiKey, systemPrompt, userContent)
}

// newDefaultMeetingManager constructs a manager with predefined agents.
func newDefaultMeetingManager() *MeetingManager {
	agents := []MeetingAgent{
		&gptAgent{cfg: AgentConfig{
			Name:        "慎重派AI",
			RoleSummary: "リスクを重視し、失敗や副作用、長期的な影響に敏感に反応する慎重なアナリスト。",
		}},
		&gptAgent{cfg: AgentConfig{
			Name:        "攻め派AI",
			RoleSummary: "短期から中期の成果を最大化することを重視し、大胆な提案を行う推進役。",
		}},
		&gptAgent{cfg: AgentConfig{
			Name:        "コンプラAI",
			RoleSummary: "法令・社内規定・倫理面からの問題を指摘し、ガバナンスを守る役割。",
		}},
		&gptAgent{cfg: AgentConfig{
			Name:        "現場AI",
			RoleSummary: "現場オペレーションや実務担当者の視点から、実現可能性・負荷を評価する役割。",
		}},
	}

	return &MeetingManager{
		Agents:    agents,
		MaxRounds: 5,
		MaxSilent: 2,
	}
}

func newRagStore() *ragStore {
	return &ragStore{
		chunks:  make([]storedChunk, 0),
		docs:    make([]document, 0),
		nextDoc: 1,
	}
}

// cosine similarity (assuming both vectors are L2-normalized)
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func sqrt(x float32) float32 {
	// simple Newton iteration, ok for our purpose
	z := x
	if z == 0 {
		return 0
	}
	for i := 0; i < 10; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}

// normalize vector in-place to unit length (L2)
func normalize(v []float32) {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	if norm == 0 {
		return
	}
	norm = sqrt(norm)
	for i, x := range v {
		v[i] = x / norm
	}
}

// --- SQLite helpers ---

func floatsToBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func bytesToFloats(b []byte) []float32 {
	if len(b)%4 != 0 {
		return nil
	}
	n := len(b) / 4
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		u := binary.LittleEndian.Uint32(b[i*4:])
		v[i] = math.Float32frombits(u)
	}
	return v
}

func initSQLite(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, err
	}
	// conservative pragmas; keep it simple
	if _, err := db.Exec(`PRAGMA journal_mode=WAL;`); err != nil {
		return nil, err
	}
	if _, err := db.Exec(`PRAGMA synchronous=NORMAL;`); err != nil {
		return nil, err
	}
	schema := `
CREATE TABLE IF NOT EXISTS documents (
	id INTEGER PRIMARY KEY,
	content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
	id INTEGER PRIMARY KEY,
	doc_id INTEGER NOT NULL,
	text TEXT NOT NULL,
	embedding BLOB NOT NULL,
	FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
`
	if _, err := db.Exec(schema); err != nil {
		return nil, err
	}
	return db, nil
}

func loadFromSQLite(db *sql.DB, store *ragStore) error {
	rows, err := db.Query(`SELECT id, content FROM documents ORDER BY id`)
	if err != nil {
		return err
	}
	defer rows.Close()

	var docs []document
	var maxID int
	for rows.Next() {
		var id int
		var content string
		if err := rows.Scan(&id, &content); err != nil {
			return err
		}
		docs = append(docs, document{
			ID:      id,
			Content: content,
			// Chunks will be filled from chunks table as needed if required
		})
		if id > maxID {
			maxID = id
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}

	cRows, err := db.Query(`SELECT doc_id, text, embedding FROM chunks ORDER BY id`)
	if err != nil {
		return err
	}
	defer cRows.Close()

	var chunks []storedChunk
	for cRows.Next() {
		var docID int
		var text string
		var emb []byte
		if err := cRows.Scan(&docID, &text, &emb); err != nil {
			return err
		}
		vec := bytesToFloats(emb)
		if vec == nil {
			continue
		}
		chunks = append(chunks, storedChunk{
			DocID:  docID,
			Text:   text,
			Vector: vec,
		})
	}
	if err := cRows.Err(); err != nil {
		return err
	}

	store.mu.Lock()
	defer store.mu.Unlock()
	store.docs = docs
	store.chunks = chunks
	store.nextDoc = maxID + 1
	return nil
}

func saveDocumentSQLite(db *sql.DB, docID int, content string, chunks []string, vectors [][]float32) error {
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	if _, err = tx.Exec(`INSERT INTO documents (id, content) VALUES (?, ?)`, docID, content); err != nil {
		return err
	}
	stmt, err := tx.Prepare(`INSERT INTO chunks (doc_id, text, embedding) VALUES (?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for i, ch := range chunks {
		if _, err = stmt.Exec(docID, ch, floatsToBytes(vectors[i])); err != nil {
			return err
		}
	}

	if err = tx.Commit(); err != nil {
		return err
	}
	return nil
}

// API request/response types
type ingestRequest struct {
	Content string `json:"content"`
}

type ingestResponse struct {
	DocumentID int `json:"document_id"`
}

type queryRequest struct {
	Query        string `json:"query"`
	TopK         int    `json:"top_k,omitempty"`
	SystemPrompt string `json:"system_prompt,omitempty"`
}

type queryResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
}

// Meeting API request/response types
type meetingRequest struct {
	Topic     string `json:"topic"`
	MaxRounds int    `json:"max_rounds,omitempty"`
}

type meetingResponse struct {
	Topic   string    `json:"topic"`
	Logs    []Message `json:"logs"`
	Summary string    `json:"summary"`
}

// OpenAI HTTP client helpers
const (
	embeddingModel = "text-embedding-3-small"
	chatModel      = "gpt-4.1-mini"
)

type openAIEmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message openAIChatMessage `json:"message"`
	} `json:"choices"`
}

func createEmbedding(ctx context.Context, apiKey string, text string) ([]float32, error) {
	reqBody := openAIEmbeddingRequest{
		Model: embeddingModel,
		Input: []string{text},
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/embeddings", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		var bodyBytes []byte
		bodyBytes, _ = io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var parsed openAIEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}
	if len(parsed.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}
	return parsed.Data[0].Embedding, nil
}

func chatCompletion(ctx context.Context, apiKey, systemPrompt, userContent string) (string, error) {
	reqBody := openAIChatRequest{
		Model: chatModel,
		Messages: []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userContent},
		},
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("chat API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var parsed openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}
	return parsed.Choices[0].Message.Content, nil
}

// handleMeeting handles virtual multi-agent meeting requests.
func handleMeeting(apiKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req meetingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Topic) == "" {
			http.Error(w, "topic is required", http.StatusBadRequest)
			return
		}

		ctx := context.Background()
		manager := newDefaultMeetingManager()
		if req.MaxRounds > 0 {
			manager.MaxRounds = req.MaxRounds
		}

		result, err := manager.RunMeeting(ctx, apiKey, req.Topic)
		if err != nil {
			log.Printf("meeting error: %v", err)
			http.Error(w, "meeting error", http.StatusInternalServerError)
			return
		}

		resp := meetingResponse{
			Topic:   result.Topic,
			Logs:    result.Logs,
			Summary: result.Summary,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(&resp)
	}
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	store := newRagStore()

	// Optional SQLite persistence (rag.db in current directory)
	var db *sql.DB
	sqlitePath := os.Getenv("RAG_SQLITE_PATH")
	if sqlitePath == "" {
		sqlitePath = "rag.db"
	}
	db, err := initSQLite(sqlitePath)
	if err != nil {
		log.Fatalf("failed to init sqlite: %v", err)
	}
	defer db.Close()

	if err := loadFromSQLite(db, store); err != nil {
		log.Printf("failed to load from sqlite (continuing with empty store): %v", err)
	}

	mux := http.NewServeMux()

	// Simple web UI
	mux.Handle("/", http.FileServer(http.Dir("static")))

	// Multi-agent virtual meeting API
	mux.HandleFunc("/meeting", handleMeeting(apiKey))

	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	// Ingest document (plain text)
	mux.HandleFunc("/documents", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ingestRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if req.Content == "" {
			http.Error(w, "content is required", http.StatusBadRequest)
			return
		}

		// simple splitter: split by ~800 characters
		const chunkSize = 800
		var chunks []string
		runes := []rune(req.Content)
		for i := 0; i < len(runes); i += chunkSize {
			end := i + chunkSize
			if end > len(runes) {
				end = len(runes)
			}
			chunks = append(chunks, string(runes[i:end]))
		}

		ctx := context.Background()
		// create embeddings for each chunk
		vectors := make([][]float32, 0, len(chunks))
		for _, ch := range chunks {
			vec, err := createEmbedding(ctx, apiKey, ch)
			if err != nil {
				log.Printf("embedding error: %v", err)
				http.Error(w, "embedding error", http.StatusInternalServerError)
				return
			}
			normalize(vec)
			vectors = append(vectors, vec)
		}

		// store in memory
		store.mu.Lock()
		docID := store.nextDoc
		store.nextDoc++
		doc := document{
			ID:      docID,
			Content: req.Content,
			Chunks:  chunks,
		}
		store.docs = append(store.docs, doc)
		for i, ch := range chunks {
			store.chunks = append(store.chunks, storedChunk{
				DocID:  docID,
				Text:   ch,
				Vector: vectors[i],
			})
		}
		store.mu.Unlock()

		// store in sqlite (best-effort; errors are logged but don't fail the request)
		if err := saveDocumentSQLite(db, docID, req.Content, chunks, vectors); err != nil {
			log.Printf("failed to persist to sqlite: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(&ingestResponse{DocumentID: docID})
	})

	// Ask question with RAG
	mux.HandleFunc("/query", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req queryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if req.Query == "" {
			http.Error(w, "query is required", http.StatusBadRequest)
			return
		}
		if req.TopK <= 0 {
			req.TopK = 3
		}

		ctx := context.Background()

		// embed query
		queryVec, err := createEmbedding(ctx, apiKey, req.Query)
		if err != nil {
			log.Printf("embedding error: %v", err)
			http.Error(w, "embedding error", http.StatusInternalServerError)
			return
		}
		normalize(queryVec)

		// rank chunks
		store.mu.RLock()
		defer store.mu.RUnlock()
		if len(store.chunks) == 0 {
			http.Error(w, "no documents ingested yet", http.StatusBadRequest)
			return
		}

		type scored struct {
			ch storedChunk
			s  float32
		}
		scoredChunks := make([]scored, 0, len(store.chunks))
		for _, ch := range store.chunks {
			score := cosineSimilarity(queryVec, ch.Vector)
			scoredChunks = append(scoredChunks, scored{
				ch: ch,
				s:  score,
			})
		}

		// partial sort: naive O(n log n) for simplicity
		sort.Slice(scoredChunks, func(i, j int) bool {
			return scoredChunks[i].s > scoredChunks[j].s
		})

		if req.TopK > len(scoredChunks) {
			req.TopK = len(scoredChunks)
		}

		contextTexts := make([]string, 0, req.TopK)
		for i := 0; i < req.TopK; i++ {
			contextTexts = append(contextTexts, scoredChunks[i].ch.Text)
		}

		// build system prompt + context
		systemPrompt := req.SystemPrompt
		if systemPrompt == "" {
			systemPrompt = "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say you don't know."
		}

		// join contexts
		fullContext := ""
		for i, t := range contextTexts {
			fullContext += "【Chunk " + strconv.Itoa(i+1) + "】\n" + t + "\n\n"
		}

		answer, err := chatCompletion(ctx, apiKey, systemPrompt, "Context:\n"+fullContext+"\n\nQuestion:\n"+req.Query)
		if err != nil {
			log.Printf("chat error: %v", err)
			http.Error(w, "chat error", http.StatusInternalServerError)
			return
		}

		resp := queryResponse{
			Answer:  answer,
			Sources: contextTexts,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(&resp)
	})

	log.Println("Server starting on :8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatalf("failed to start server: %v", err)
	}
}
