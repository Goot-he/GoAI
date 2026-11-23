package rag

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"
)

// ---------- Embedder ----------

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
	Name() string
}

// HashEmbedder 是不依赖外部 API 的确定性向量化实现。
// 使用中文 uni/bigram + ASCII 词项哈希，适合演示与无 Key 场景；
// 语义能力弱于真实 embedding，因此检索默认与 BM25 混合。
type HashEmbedder struct{ dim int }

func NewHashEmbedder(dim int) *HashEmbedder {
	if dim <= 0 {
		dim = 384
	}
	return &HashEmbedder{dim: dim}
}

func (h *HashEmbedder) Dim() int     { return h.dim }
func (h *HashEmbedder) Name() string { return "hash" }

func (h *HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.embedOne(t)
	}
	return out, nil
}

func (h *HashEmbedder) embedOne(text string) []float32 {
	vec := make([]float32, h.dim)
	for _, tok := range Tokenize(text) {
		h1 := fnv.New64a()
		_, _ = h1.Write([]byte(tok))
		idx := int(h1.Sum64() % uint64(h.dim))

		h2 := fnv.New64a()
		_, _ = h2.Write([]byte(tok))
		sign := 1.0
		if h2.Sum64()%2 == 1 {
			sign = -1.0
		}
		weight := 1.0
		if len([]rune(tok)) > 2 {
			weight = 0.5
		}
		vec[idx] += float32(sign * weight)
	}
	return normalize(vec)
}

// OpenAIEmbedder 走标准 /embeddings 接口，兼容 OpenAI、DashScope 兼容模式、
// 硅基流动等提供 embedding 的服务。
type OpenAIEmbedder struct {
	baseURL, apiKey, model string
	dim                    int
	client                 *http.Client
}

func NewOpenAIEmbedder(baseURL, apiKey, model string, dim int) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		dim:     dim,
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (o *OpenAIEmbedder) Dim() int     { return o.dim }
func (o *OpenAIEmbedder) Name() string { return "openai:" + o.model }

func (o *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(map[string]any{"model": o.model, "input": texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding 请求失败: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("embedding 返回 %s: %s", resp.Status, string(raw))
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embedding 响应解析失败: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embedding 数量不匹配: 期望 %d 实际 %d", len(texts), len(out.Data))
	}
	vecs := make([][]float32, len(out.Data))
	for i, d := range out.Data {
		if o.dim > 0 && len(d.Embedding) != o.dim {
			return nil, fmt.Errorf("embedding 维度不匹配: 期望 %d 实际 %d", o.dim, len(d.Embedding))
		}
		if o.dim == 0 {
			o.dim = len(d.Embedding)
		}
		vecs[i] = normalize(d.Embedding)
	}
	return vecs, nil
}

// ---------- 向量编解码 ----------

func EncodeVector(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func DecodeVector(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	if sum == 0 {
		return v
	}
	n := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= n
	}
	return v
}

// Cosine 计算两个已归一化向量的余弦相似度。
func Cosine(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// ---------- 分词（中文 bigram + ASCII 词） ----------

var asciiWord = regexp.MustCompile(`[a-z0-9_]+`)

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	var out []string
	var hanBuf []rune

	flushHan := func() {
		if len(hanBuf) == 0 {
			return
		}
		for _, r := range hanBuf {
			out = append(out, string(r))
		}
		for i := 0; i+1 < len(hanBuf); i++ {
			out = append(out, string(hanBuf[i:i+2]))
		}
		hanBuf = hanBuf[:0]
	}

	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			hanBuf = append(hanBuf, r)
			continue
		}
		flushHan()
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			out = append(out, string(r))
		}
	}
	flushHan()

	// ASCII 连续字母数字合并成词，提升英文/型号匹配
	words := asciiWord.FindAllString(text, -1)
	for _, w := range words {
		if len(w) >= 3 {
			out = append(out, w)
		}
	}
	return out
}

// ---------- 文档解析 ----------

var ErrUnsupportedFile = errors.New("仅支持 .txt / .md / .markdown / .docx 文档")

func ExtractText(name string, data []byte) (string, error) {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".txt", ".md", ".markdown", ".log", ".csv":
		return normalizeNewlines(string(data)), nil
	case ".docx":
		return extractDocx(data)
	default:
		return "", ErrUnsupportedFile
	}
}

func normalizeNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	return strings.TrimSpace(s)
}

// extractDocx 只依赖标准库：解压 word/document.xml，把段落与换行还原为纯文本。
func extractDocx(data []byte) (string, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", fmt.Errorf("docx 解析失败: %w", err)
	}
	var doc *zip.File
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			doc = f
			break
		}
	}
	if doc == nil {
		return "", fmt.Errorf("docx 缺少 word/document.xml")
	}
	rc, err := doc.Open()
	if err != nil {
		return "", err
	}
	defer rc.Close()
	raw, err := io.ReadAll(io.LimitReader(rc, 32<<20))
	if err != nil {
		return "", err
	}

	xmlText := string(raw)
	xmlText = strings.ReplaceAll(xmlText, "</w:p>", "\n")
	xmlText = strings.ReplaceAll(xmlText, "<w:br/>", "\n")
	xmlText = strings.ReplaceAll(xmlText, "<w:tab/>", "\t")
	// 去除所有标签
	tagRe := regexp.MustCompile(`<[^>]+>`)
	xmlText = tagRe.ReplaceAllString(xmlText, "")
	xmlText = strings.ReplaceAll(xmlText, "&lt;", "<")
	xmlText = strings.ReplaceAll(xmlText, "&gt;", ">")
	xmlText = strings.ReplaceAll(xmlText, "&amp;", "&")
	xmlText = strings.ReplaceAll(xmlText, "&quot;", `"`)
	xmlText = strings.ReplaceAll(xmlText, "&apos;", "'")
	return normalizeNewlines(xmlText), nil
}

// ---------- 递归切分 ----------

type Chunker struct {
	Size       int
	Overlap    int
	Separators []string
}

func NewChunker(size, overlap int) *Chunker {
	return &Chunker{
		Size:       size,
		Overlap:    overlap,
		Separators: []string{"\n\n", "\n", "。", "！", "？", "；", ".", " ", ""},
	}
}

func (c *Chunker) Split(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	chunks := c.splitRecursive(text, 0)
	chunks = c.withOverlap(chunks)
	// 过滤过短碎片
	out := make([]string, 0, len(chunks))
	for _, ch := range chunks {
		if len([]rune(strings.TrimSpace(ch))) >= 8 {
			out = append(out, strings.TrimSpace(ch))
		}
	}
	return out
}

func (c *Chunker) splitRecursive(text string, sepIdx int) []string {
	if len([]rune(text)) <= c.Size {
		return []string{text}
	}
	if sepIdx >= len(c.Separators) {
		return hardSplit([]rune(text), c.Size)
	}
	sep := c.Separators[sepIdx]
	if sep == "" {
		return hardSplit([]rune(text), c.Size)
	}
	parts := strings.Split(text, sep)
	var atoms []string
	for _, p := range parts {
		if len([]rune(p)) > c.Size {
			atoms = append(atoms, c.splitRecursive(p, sepIdx+1)...)
			continue
		}
		if strings.TrimSpace(p) != "" {
			atoms = append(atoms, p)
		}
	}
	return mergeAtoms(atoms, sep, c.Size)
}

func mergeAtoms(atoms []string, sep string, size int) []string {
	var out []string
	var cur strings.Builder
	curLen := 0
	flush := func() {
		if strings.TrimSpace(cur.String()) != "" {
			out = append(out, cur.String())
		}
		cur.Reset()
		curLen = 0
	}
	for _, a := range atoms {
		a = strings.TrimSpace(a)
		if a == "" {
			continue
		}
		l := len([]rune(a))
		need := l
		if curLen > 0 {
			need = l + len([]rune(sep))
		}
		if curLen > 0 && curLen+need > size {
			flush()
		}
		if curLen > 0 {
			cur.WriteString(sep)
		}
		cur.WriteString(a)
		curLen += need
	}
	flush()
	return out
}

func hardSplit(runes []rune, size int) []string {
	var out []string
	for i := 0; i < len(runes); i += size {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
	}
	return out
}

func (c *Chunker) withOverlap(chunks []string) []string {
	if c.Overlap <= 0 || len(chunks) <= 1 {
		return chunks
	}
	out := make([]string, 0, len(chunks))
	for i, ch := range chunks {
		if i == 0 {
			out = append(out, ch)
			continue
		}
		prevRunes := []rune(chunks[i-1])
		start := len(prevRunes) - c.Overlap
		if start < 0 {
			start = 0
		}
		out = append(out, string(prevRunes[start:])+"\n"+ch)
	}
	return out
}
