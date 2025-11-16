package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
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

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	store := newRagStore()

	mux := http.NewServeMux()

	// Simple web UI
	mux.Handle("/", http.FileServer(http.Dir("static")))

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

		// store
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
