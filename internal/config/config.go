package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config 汇总服务运行所需的全部外部依赖配置。
// 遵循课程原则：敏感配置只从环境变量读取，默认值仅用于本地开发。
type Config struct {
	Addr string

	MySQLDSN string

	RedisAddr     string
	RedisPassword string
	RedisDB       int

	JWTSecret string
	JWTExpire time.Duration

	LLMBaseURL string
	LLMAPIKey  string
	LLMModel   string
	LLMTimeout time.Duration

	EmbeddingBaseURL string
	EmbeddingAPIKey  string
	EmbeddingModel   string
	EmbeddingDim     int

	QdrantURL        string
	QdrantAPIKey     string
	QdrantCollection string

	MaxIterations int
	MaxSteps      int
	ChunkSize     int
	ChunkOverlap  int
	RetrieveTopK  int
	HistoryKeep   int
	SummaryTokens int
}

func Load() (*Config, error) {
	c := &Config{
		Addr: env("HTTP_ADDR", ":8080"),

		MySQLDSN: env("MYSQL_DSN", "root:root@tcp(127.0.0.1:3307)/agent_rag?parseTime=true&charset=utf8mb4&loc=Local"),

		RedisAddr:     env("REDIS_ADDR", "127.0.0.1:6379"),
		RedisPassword: env("REDIS_PASSWORD", ""),
		RedisDB:       envInt("REDIS_DB", 0),

		JWTSecret: env("JWT_SECRET", "dev-only-secret-change-me"),
		JWTExpire: time.Duration(envInt("JWT_EXPIRE_HOURS", 168)) * time.Hour,

		LLMBaseURL: strings.TrimRight(env("LLM_BASE_URL", "https://api.deepseek.com"), "/"),
		LLMAPIKey:  env("LLM_API_KEY", ""),
		LLMModel:   env("LLM_MODEL", "deepseek-chat"),
		LLMTimeout: time.Duration(envInt("LLM_TIMEOUT_SECONDS", 120)) * time.Second,

		EmbeddingBaseURL: strings.TrimRight(env("EMBEDDING_BASE_URL", ""), "/"),
		EmbeddingAPIKey:  env("EMBEDDING_API_KEY", ""),
		EmbeddingModel:   env("EMBEDDING_MODEL", ""),
		EmbeddingDim:     envInt("EMBEDDING_DIM", 0),

		QdrantURL:        strings.TrimRight(env("QDRANT_URL", ""), "/"),
		QdrantAPIKey:     env("QDRANT_API_KEY", ""),
		QdrantCollection: env("QDRANT_COLLECTION", "agent_rag_chunks"),

		MaxIterations: envInt("AGENT_MAX_ITERATIONS", 10),
		MaxSteps:      envInt("AGENT_MAX_STEPS", 10),
		ChunkSize:     envInt("CHUNK_SIZE", 500),
		ChunkOverlap:  envInt("CHUNK_OVERLAP", 80),
		RetrieveTopK:  envInt("RETRIEVE_TOP_K", 5),
		HistoryKeep:   envInt("HISTORY_KEEP", 12),
		SummaryTokens: envInt("SUMMARY_TRIGGER_TOKENS", 6000),
	}

	if c.LLMAPIKey == "" {
		return nil, fmt.Errorf("缺少 LLM_API_KEY（DeepSeek API Key）")
	}
	if len(c.JWTSecret) < 16 {
		return nil, fmt.Errorf("JWT_SECRET 至少 16 个字符")
	}
	if c.ChunkSize < 100 || c.ChunkOverlap < 0 || c.ChunkOverlap >= c.ChunkSize {
		return nil, fmt.Errorf("切分参数不合法: size=%d overlap=%d", c.ChunkSize, c.ChunkOverlap)
	}
	if c.RetrieveTopK < 1 || c.RetrieveTopK > 20 {
		return nil, fmt.Errorf("RETRIEVE_TOP_K 需在 1..20")
	}
	return c, nil
}

func (c *Config) EmbeddingEnabled() bool {
	return c.EmbeddingBaseURL != "" && c.EmbeddingAPIKey != "" && c.EmbeddingModel != ""
}

// QdrantEnabled 配置了 Qdrant 地址即启用向量库检索；
// 未启用时退化为纯 BM25 关键词检索。
func (c *Config) QdrantEnabled() bool {
	return c.QdrantURL != "" && c.QdrantCollection != ""
}

func env(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
