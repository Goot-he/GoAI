package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cloudwego/eino-ext/components/model/openai"

	"agentragract/internal/agent"
	"agentragract/internal/cache"
	"agentragract/internal/config"
	"agentragract/internal/httpapi"
	"agentragract/internal/rag"
	"agentragract/internal/store"
)

// main 是服务入口：加载配置、连接 MySQL/Redis、装配 Embedder 与 Qdrant、
// 创建 ChatModel/Agent/Router，并启动 HTTP 服务、监听退出信号做优雅关闭。
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("配置错误: %v", err)
	}

	st, err := store.New(ctx, cfg)
	if err != nil {
		log.Fatalf("初始化 MySQL 失败: %v", err)
	}
	defer st.Close()
	log.Printf("MySQL 已连接并完成建表")

	c := cache.New(ctx, cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	defer c.Close()
	if c.Available() {
		log.Printf("Redis 已连接: %s", cfg.RedisAddr)
	} else {
		log.Printf("Redis 不可用，已降级为进程内缓存")
	}

	var embedder rag.Embedder
	if cfg.EmbeddingEnabled() {
		embedder = rag.NewOpenAIEmbedder(cfg.EmbeddingBaseURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel, cfg.EmbeddingDim)
		log.Printf("使用远程 Embedding: %s", embedder.Name())
	} else {
		embedder = rag.NewHashEmbedder(384)
		log.Printf("未配置 EMBEDDING_*，使用内置 Hash Embedding（向量检索 + BM25 混合）")
	}

	var qdrant *rag.QdrantStore
	if cfg.QdrantEnabled() {
		qs := rag.NewQdrantStore(cfg.QdrantURL, cfg.QdrantCollection, cfg.QdrantAPIKey, embedder.Dim())
		if err := qs.EnsureCollection(ctx); err != nil {
			log.Printf("Qdrant 初始化失败，降级为纯 BM25 检索: %v", err)
		} else {
			qdrant = qs
			log.Printf("Qdrant 已连接: %s / %s", cfg.QdrantURL, cfg.QdrantCollection)
		}
	} else {
		log.Printf("未配置 QDRANT_URL，向量检索关闭，仅使用 BM25 关键词检索")
	}

	ragSvc := rag.NewService(st, embedder, qdrant, rag.NewChunker(cfg.ChunkSize, cfg.ChunkOverlap), c, cfg.RetrieveTopK)
	if qdrant != nil {
		if err := ragSvc.Backfill(ctx); err != nil {
			log.Printf("历史数据回填失败（不影响新文档入库）: %v", err)
		}
	}

	chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		APIKey:  cfg.LLMAPIKey,
		BaseURL: cfg.LLMBaseURL,
		Model:   cfg.LLMModel,
		Timeout: cfg.LLMTimeout,
	})
	if err != nil {
		log.Fatalf("初始化 ChatModel 失败: %v", err)
	}

	ag := agent.New(cfg, st, ragSvc, chatModel)
	router := httpapi.NewRouter(cfg, st, ragSvc, ag, c.Available())

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		// 不设置 WriteTimeout：SSE 是长连接，整体超时交给请求 context。
	}

	go func() {
		log.Printf("服务已启动: http://localhost%s  (model=%s, iterations=%d)",
			cfg.Addr, cfg.LLMModel, cfg.MaxIterations)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("HTTP 服务异常退出: %v", err)
		}
	}()

	<-ctx.Done()
	log.Printf("收到退出信号，开始优雅关闭…")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("优雅关闭失败: %v", err)
	}
	log.Printf("已退出")
}
