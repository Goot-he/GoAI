package httpapi

import (
	"io/fs"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"agentragract/internal/agent"
	"agentragract/internal/config"
	"agentragract/internal/rag"
	"agentragract/internal/store"
	"agentragract/internal/webui"
)

// NewRouter 组装路由：/api 分组、JWT 保护的业务接口、健康检查，
// 以及内嵌的前端静态资源（index.html / app.js / app.css）。
func NewRouter(cfg *config.Config, st *store.Store, ragSvc *rag.Service, ag *agent.Service, cacheEnabled bool) *gin.Engine {
	a := &API{cfg: cfg, st: st, rag: ragSvc, agent: ag, cacheEnabled: cacheEnabled}

	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.ReleaseMode)
	}
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())
	r.MaxMultipartMemory = 12 << 20

	api := r.Group("/api")
	{
		api.POST("/auth/register", a.Register)
		api.POST("/auth/login", a.Login)
		api.GET("/health", a.Health)

		auth := api.Group("")
		auth.Use(a.authRequired())
		{
			auth.GET("/auth/me", a.Me)
			auth.GET("/stats", a.Stats)

			auth.GET("/conversations", a.ListConversations)
			auth.POST("/conversations", a.CreateConversation)
			auth.GET("/conversations/:id/messages", a.GetMessages)
			auth.PATCH("/conversations/:id", a.RenameConversation)
			auth.DELETE("/conversations/:id", a.DeleteConversation)

			auth.POST("/chat", a.Chat)
			auth.POST("/chat/stream", a.ChatStream)

			auth.POST("/documents", a.UploadDocument)
			auth.GET("/documents", a.ListDocuments)
			auth.DELETE("/documents/:id", a.DeleteDocument)
		}
	}

	webFS, err := fs.Sub(webui.Content, ".")
	if err != nil {
		panic(err)
	}

	// 直接读取内嵌文件返回，避免 http.FileServer 对 index.html 的 301 重定向
	serve := func(name, contentType string) gin.HandlerFunc {
		return func(c *gin.Context) {
			data, err := fs.ReadFile(webFS, name)
			if err != nil {
				c.String(http.StatusNotFound, "not found")
				return
			}
			c.Data(http.StatusOK, contentType, data)
		}
	}
	r.GET("/", serve("index.html", "text/html; charset=utf-8"))
	r.GET("/app.js", serve("app.js", "application/javascript; charset=utf-8"))
	r.GET("/app.css", serve("app.css", "text/css; charset=utf-8"))
	r.NoRoute(a.NotFound)

	return r
}

// orDefault 在字符串为空时返回默认值。
func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
