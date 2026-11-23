package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"agentragract/internal/store"
)

// QdrantStore 通过 Qdrant REST API 存取向量。
// point id 直接使用 MySQL chunks.id，正文仍以 MySQL 为准，
// payload 只保存过滤与展示所需的元数据。
type QdrantStore struct {
	baseURL    string
	collection string
	apiKey     string
	dim        int
	client     *http.Client
}

func NewQdrantStore(baseURL, collection, apiKey string, dim int) *QdrantStore {
	return &QdrantStore{
		baseURL:    baseURL,
		collection: collection,
		apiKey:     apiKey,
		dim:        dim,
		client:     &http.Client{Timeout: 60 * time.Second},
	}
}

func (q *QdrantStore) Name() string { return "qdrant:" + q.collection }

// EnsureCollection 创建集合（若不存在），并建立过滤字段索引。
func (q *QdrantStore) EnsureCollection(ctx context.Context) error {
	if q.dim <= 0 {
		return fmt.Errorf("Qdrant 需要已知向量维度，请配置 EMBEDDING_DIM（nomic-embed-text 为 768）")
	}
	// do 对非 2xx 也会返回状态码，因此先判 404（集合不存在），再判其他错误
	status, _, err := q.do(ctx, http.MethodGet, "/collections/"+q.collection, nil)
	if status == http.StatusNotFound {
		body := map[string]any{
			"vectors": map[string]any{"size": q.dim, "distance": "Cosine"},
		}
		if _, _, err := q.do(ctx, http.MethodPut, "/collections/"+q.collection, body); err != nil {
			return fmt.Errorf("创建 Qdrant 集合失败: %w", err)
		}
		log.Printf("Qdrant 集合已创建: %s (dim=%d, distance=Cosine)", q.collection, q.dim)
	} else if err != nil {
		return err
	}

	// 过滤字段索引：按用户隔离、按文档删除
	for _, idx := range []struct{ field, schema string }{
		{"user_id", "integer"},
		{"document_id", "keyword"},
	} {
		body := map[string]any{"field_name": idx.field, "field_schema": idx.schema}
		if _, _, err := q.do(ctx, http.MethodPut, "/collections/"+q.collection+"/index", body); err != nil {
			log.Printf("Qdrant 字段索引创建跳过(%s): %v", idx.field, err)
		}
	}
	return nil
}

// PointsCount 返回集合中的向量条数，用于判断是否需要回填历史数据。
func (q *QdrantStore) PointsCount(ctx context.Context) (int64, error) {
	_, raw, err := q.do(ctx, http.MethodGet, "/collections/"+q.collection, nil)
	if err != nil {
		return 0, err
	}
	var out struct {
		Result struct {
			PointsCount int64 `json:"points_count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, err
	}
	return out.Result.PointsCount, nil
}

// Upsert 写入/覆盖一批向量，chunks[i] 与 vecs[i] 一一对应。
func (q *QdrantStore) Upsert(ctx context.Context, chunks []store.Chunk, vecs [][]float32) error {
	if len(chunks) == 0 {
		return nil
	}
	if len(chunks) != len(vecs) {
		return fmt.Errorf("向量数量不匹配: chunks=%d vecs=%d", len(chunks), len(vecs))
	}
	points := make([]map[string]any, 0, len(chunks))
	for i, c := range chunks {
		if len(vecs[i]) == 0 {
			continue
		}
		points = append(points, map[string]any{
			"id":     c.ID,
			"vector": vecs[i],
			"payload": map[string]any{
				"user_id":     c.UserID,
				"document_id": c.DocumentID,
				"seq":         c.Seq,
				"doc_name":    c.DocName,
			},
		})
	}
	if len(points) == 0 {
		return nil
	}
	_, _, err := q.do(ctx, http.MethodPut, "/collections/"+q.collection+"/points?wait=true",
		map[string]any{"points": points})
	return err
}

// DeleteDocument 删除某篇文档的全部向量。
func (q *QdrantStore) DeleteDocument(ctx context.Context, docID string) error {
	body := map[string]any{
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "document_id", "match": map[string]any{"value": docID}},
			},
		},
	}
	_, _, err := q.do(ctx, http.MethodPost, "/collections/"+q.collection+"/points/delete?wait=true", body)
	return err
}

// Search 在指定用户的向量空间内做 ANN 检索，返回 chunkID -> 余弦得分。
func (q *QdrantStore) Search(ctx context.Context, userID int64, vec []float32, limit int) (map[int64]float64, error) {
	out := make(map[int64]float64, limit)
	if len(vec) == 0 {
		return out, nil
	}
	body := map[string]any{
		"vector": vec,
		"limit":  limit,
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "user_id", "match": map[string]any{"value": userID}},
			},
		},
		"with_payload": false,
		"with_vector":  false,
	}
	_, raw, err := q.do(ctx, http.MethodPost, "/collections/"+q.collection+"/points/search", body)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Result []struct {
			ID    int64   `json:"id"`
			Score float64 `json:"score"`
		} `json:"result"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("Qdrant 检索响应解析失败: %w", err)
	}
	for _, r := range resp.Result {
		if r.Score > 0 {
			out[r.ID] = r.Score
		}
	}
	return out, nil
}

// do 统一发起请求并返回状态码与响应体；非 2xx 时附带服务端错误文本。
func (q *QdrantStore) do(ctx context.Context, method, path string, body any) (int, []byte, error) {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		reader = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.baseURL+path, reader)
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}
	resp, err := q.client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("Qdrant 请求失败(%s %s): %w", method, path, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return resp.StatusCode, nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, raw, fmt.Errorf("Qdrant %s %s 返回 %s: %s",
			method, path, resp.Status, string(raw[:min(len(raw), 1024)]))
	}
	return resp.StatusCode, raw, nil
}
