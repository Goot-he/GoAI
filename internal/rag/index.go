package rag

import (
	"context"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"agentragract/internal/cache"
	"agentragract/internal/store"
)

// Hit 是一条检索结果。
type Hit struct {
	DocumentID string  `json:"document_id"`
	DocName    string  `json:"doc_name"`
	Seq        int     `json:"seq"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
}

// indexedItem 是 BM25 索引中的一条片段：原始片段、词频表与词数。
type indexedItem struct {
	chunk  store.Chunk
	tf     map[string]int
	length int
}

// userIndex 是单个用户的 BM25 索引；version 变化会让检索缓存自动失效。
type userIndex struct {
	version int64
	items   []*indexedItem
	df      map[string]int
	avgLen  float64
}

// Service 负责文档入库与用户级混合检索。
// 向量召回由 Qdrant（HNSW）承担，BM25 由进程内索引承担，
// 两路结果各自归一化后加权融合。
type Service struct {
	store    *store.Store
	embedder Embedder
	vectors  *QdrantStore
	chunker  *Chunker
	cache    *cache.Cache
	topK     int

	mu      sync.Mutex
	indexes map[int64]*userIndex
}

// NewService 构造 RAG 服务，向量库可为 nil（此时退化为纯 BM25）。
func NewService(st *store.Store, emb Embedder, vectors *QdrantStore, chunker *Chunker, c *cache.Cache, topK int) *Service {
	return &Service{
		store:    st,
		embedder: emb,
		vectors:  vectors,
		chunker:  chunker,
		cache:    c,
		topK:     topK,
		indexes:  make(map[int64]*userIndex),
	}
}

// EmbedderName 返回当前 Embedder 标识，用于缓存键与前端统计展示。
func (s *Service) EmbedderName() string { return s.embedder.Name() }

// VectorStoreName 返回当前向量库标识，未启用时返回 "disabled(BM25 only)"。
func (s *Service) VectorStoreName() string {
	if s.vectors == nil {
		return "disabled(BM25 only)"
	}
	return s.vectors.Name()
}

// Ingest 解析、切分后写 MySQL（正文），向量经 Embedder 计算后写入 Qdrant。
// 同步执行，前端轮询/刷新即可看到状态。
func (s *Service) Ingest(ctx context.Context, userID int64, docID, name string, size int64, data []byte) (int, error) {
	text, err := ExtractText(name, data)
	if err != nil {
		return 0, err
	}
	if strings.TrimSpace(text) == "" {
		return 0, fmt.Errorf("文档内容为空")
	}
	chunks := s.chunker.Split(text)
	if len(chunks) == 0 {
		return 0, fmt.Errorf("文档切分后没有有效内容")
	}

	rows := make([]store.Chunk, 0, len(chunks))
	for i, ch := range chunks {
		rows = append(rows, store.Chunk{
			DocumentID: docID,
			UserID:     userID,
			Seq:        i,
			DocName:    name,
			Content:    ch,
		})
	}
	ids, err := s.store.InsertChunks(ctx, rows)
	if err != nil {
		return 0, err
	}
	for i := range rows {
		rows[i].ID = ids[i]
	}

	if s.vectors != nil {
		if err := s.embedAndUpsert(ctx, rows, chunks); err != nil {
			return 0, fmt.Errorf("向量入库失败（正文已保存，可重新上传或重启回填）: %w", err)
		}
	}
	s.Invalidate(userID)
	return len(rows), nil
}

// embedAndUpsert 分批向量化并写入 Qdrant。
func (s *Service) embedAndUpsert(ctx context.Context, rows []store.Chunk, texts []string) error {
	const batch = 16
	for start := 0; start < len(rows); start += batch {
		end := min(start+batch, len(rows))
		vecs, err := s.embedder.Embed(ctx, texts[start:end])
		if err != nil {
			return fmt.Errorf("向量化失败: %w", err)
		}
		if err := s.vectors.Upsert(ctx, rows[start:end], vecs); err != nil {
			return err
		}
	}
	return nil
}

// RemoveDocument 在 MySQL 删除文档后，同步删除其向量并失效索引。
func (s *Service) RemoveDocument(ctx context.Context, docID string, userID int64) {
	if s.vectors != nil {
		if err := s.vectors.DeleteDocument(ctx, docID); err != nil {
			log.Printf("Qdrant 删除文档向量失败(doc=%s): %v", docID, err)
		}
	}
	s.Invalidate(userID)
}

// Backfill 在启动时把历史片段回填到向量库：
// 仅当集合为空且 MySQL 中已有片段时执行，避免每次启动重复计算。
func (s *Service) Backfill(ctx context.Context) error {
	if s.vectors == nil {
		return nil
	}
	count, err := s.vectors.PointsCount(ctx)
	if err != nil {
		return err
	}
	if count > 0 {
		return nil
	}
	chunks, err := s.store.ListAllChunks(ctx)
	if err != nil {
		return err
	}
	if len(chunks) == 0 {
		return nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Content
	}
	log.Printf("Qdrant 集合为空，开始回填 %d 条历史片段...", len(chunks))
	if err := s.embedAndUpsert(ctx, chunks, texts); err != nil {
		return err
	}
	log.Printf("历史片段回填完成: %d 条", len(chunks))
	return nil
}

// Invalidate 在文档变更后清除用户索引，下次检索重建。
func (s *Service) Invalidate(userID int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.indexes, userID)
}

// getIndex 返回用户的 BM25 索引：有缓存直接复用，否则从 MySQL 全量构建，
// 并计算词频、文档频率与平均长度。
func (s *Service) getIndex(ctx context.Context, userID int64) (*userIndex, error) {
	s.mu.Lock()
	if idx, ok := s.indexes[userID]; ok {
		s.mu.Unlock()
		return idx, nil
	}
	s.mu.Unlock()

	chunks, err := s.store.ListChunksByUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	idx := &userIndex{
		version: time.Now().UnixNano(),
		df:      make(map[string]int),
	}
	totalLen := 0
	for _, c := range chunks {
		tokens := Tokenize(c.Content)
		if len(tokens) == 0 {
			continue
		}
		tf := make(map[string]int, len(tokens))
		for _, t := range tokens {
			tf[t]++
		}
		// 文档频率按去重词计
		for t := range tf {
			idx.df[t]++
		}
		it := &indexedItem{chunk: c, tf: tf, length: len(tokens)}
		idx.items = append(idx.items, it)
		totalLen += it.length
	}
	if len(idx.items) > 0 {
		idx.avgLen = float64(totalLen) / float64(len(idx.items))
	}

	s.mu.Lock()
	s.indexes[userID] = idx
	s.mu.Unlock()
	return idx, nil
}

// Search 执行一次混合检索。
func (s *Service) Search(ctx context.Context, userID int64, query string, topK int) ([]Hit, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	if topK <= 0 {
		topK = s.topK
	}

	idx, err := s.getIndex(ctx, userID)
	if err != nil {
		return nil, err
	}
	if len(idx.items) == 0 {
		return nil, nil
	}

	cacheKey := fmt.Sprintf("rag:%d:%d:%x", userID, idx.version, hashString(fmt.Sprintf("%s|%d", query, topK)))
	var cached []Hit
	if s.cache.GetJSON(ctx, cacheKey, &cached) {
		return cached, nil
	}

	var queryVec []float32
	if s.vectors != nil {
		if qv, err := s.embedQuery(ctx, query); err == nil {
			queryVec = qv
		} else {
			log.Printf("查询向量化失败，降级为关键词检索: %v", err)
		}
	}

	const poolSize = 40
	byKey := make(map[string]*indexedItem, len(idx.items))
	byID := make(map[int64]*indexedItem, len(idx.items))
	for _, it := range idx.items {
		byKey[chunkKey(it.chunk)] = it
		byID[it.chunk.ID] = it
	}

	// 1) 向量召回：Qdrant ANN（HNSW + 余弦），按 user_id 过滤，取前 poolSize 条
	vecScores := make(map[string]float64, poolSize)
	if len(queryVec) > 0 {
		scores, err := s.vectors.Search(ctx, userID, queryVec, poolSize)
		if err != nil {
			log.Printf("Qdrant 检索失败，降级为关键词检索: %v", err)
		} else {
			for chunkID, score := range scores {
				if it, ok := byID[chunkID]; ok {
					vecScores[chunkKey(it.chunk)] = score
				}
			}
		}
	}

	// 2) 关键词召回：BM25，取前 poolSize 条
	kwScores := make(map[string]float64, poolSize)
	kw := bm25Scores(query, idx)
	sort.Slice(kw, func(i, j int) bool { return kw[i].score > kw[j].score })
	if len(kw) > poolSize {
		kw = kw[:poolSize]
	}
	for _, c := range kw {
		kwScores[c.id] = c.score
	}

	// 3) 分值级融合：两路各自按最大值归一化后加权，避免用排名把精确命中稀释掉
	var maxVec, maxKw float64
	for _, v := range vecScores {
		if v > maxVec {
			maxVec = v
		}
	}
	for _, v := range kwScores {
		if v > maxKw {
			maxKw = v
		}
	}
	fused := make(map[string]float64, len(vecScores)+len(kwScores))
	const wVec, wKw = 0.45, 0.55
	for id, v := range vecScores {
		score := 0.0
		if maxVec > 0 {
			score = wVec * v / maxVec
		}
		fused[id] += score
	}
	for id, v := range kwScores {
		score := 0.0
		if maxKw > 0 {
			score = wKw * v / maxKw
		}
		fused[id] += score
	}

	type fusedItem struct {
		id    string
		score float64
	}
	all := make([]fusedItem, 0, len(fused))
	for id, sc := range fused {
		all = append(all, fusedItem{id, sc})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	if len(all) > topK {
		all = all[:topK]
	}

	hits := make([]Hit, 0, len(all))
	for _, f := range all {
		it := byKey[f.id]
		if it == nil {
			continue
		}
		hits = append(hits, Hit{
			DocumentID: it.chunk.DocumentID,
			DocName:    it.chunk.DocName,
			Seq:        it.chunk.Seq,
			Content:    snippet(it.chunk.Content, 700),
			Score:      f.score,
		})
	}

	s.cache.SetJSON(ctx, cacheKey, hits, 10*time.Minute)
	return hits, nil
}

// embedQuery 对查询文本做向量化，结果缓存 24 小时以减少 Embedding 调用。
func (s *Service) embedQuery(ctx context.Context, query string) ([]float32, error) {
	key := "emb:" + fmt.Sprintf("%x", hashString(query)) + ":" + s.embedder.Name()
	var cached []float32
	if s.cache.GetJSON(ctx, key, &cached) && len(cached) > 0 {
		return cached, nil
	}
	vecs, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, fmt.Errorf("空向量")
	}
	s.cache.SetJSON(ctx, key, vecs[0], 24*time.Hour)
	return vecs[0], nil
}

// bm25Candidate 是 BM25 召回的候选：片段键与得分。
type bm25Candidate struct {
	id    string
	score float64
}

// bm25Scores 计算查询对各片段的 BM25 得分（k1=1.5，b=0.75）。
func bm25Scores(query string, idx *userIndex) []bm25Candidate {
	const k1, b = 1.5, 0.75
	qTokens := Tokenize(query)
	if len(qTokens) == 0 {
		return nil
	}
	n := float64(len(idx.items))
	out := make([]bm25Candidate, 0, len(idx.items))
	for _, it := range idx.items {
		var score float64
		for _, t := range dedup(qTokens) {
			tf := float64(it.tf[t])
			if tf == 0 {
				continue
			}
			df := float64(idx.df[t])
			idf := math.Log(1 + (n-df+0.5)/(df+0.5))
			score += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+b*float64(it.length)/math.Max(idx.avgLen, 1)))
		}
		if score > 0 {
			out = append(out, bm25Candidate{id: chunkKey(it.chunk), score: score})
		}
	}
	return out
}

// chunkKey 生成片段的稳定键（文档 ID + 段号），用于融合两路召回结果。
func chunkKey(c store.Chunk) string { return c.DocumentID + "#" + fmt.Sprint(c.Seq) }

// dedup 去掉查询词中的重复项，避免同一词重复累加 BM25 得分。
func dedup(tokens []string) []string {
	seen := make(map[string]struct{}, len(tokens))
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		out = append(out, t)
	}
	return out
}

// hashString 用 FNV-64a 计算字符串哈希，用于生成缓存键。
func hashString(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

// snippet 按 rune 截断片段正文，控制给模型的上下文体积。
func snippet(s string, maxRunes int) string {
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return string(r[:maxRunes]) + "…"
}

// FormatHits 把检索结果整理成给模型看的观察文本（带来源标记，便于引用）。
func FormatHits(hits []Hit) string {
	if len(hits) == 0 {
		return "知识库中未检索到相关内容。请如实告诉用户资料未涵盖，不要编造。"
	}
	var sb strings.Builder
	sb.WriteString("以下是你的知识库检索结果，仅供你组织答案时参考。不要把方括号、来源编号、片段编号等标记原样写进给用户的回答。\n\n")
	for i, h := range hits {
		fmt.Fprintf(&sb, "资料%d：出自文档《%s》第%d段，相关度 %.3f\n%s\n\n", i+1, h.DocName, h.Seq+1, h.Score, h.Content)
	}
	return sb.String()
}
