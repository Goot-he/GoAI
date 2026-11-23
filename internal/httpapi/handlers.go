package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"agentragract/internal/agent"
	"agentragract/internal/config"
	"agentragract/internal/rag"
	"agentragract/internal/store"
)

type API struct {
	cfg          *config.Config
	st           *store.Store
	rag          *rag.Service
	agent        *agent.Service
	cacheEnabled bool
}

func newUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ---------- conversations ----------

func (a *API) ListConversations(c *gin.Context) {
	list, err := a.st.ListConversations(c.Request.Context(), userIDFrom(c))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	items := make([]gin.H, 0, len(list))
	for _, cv := range list {
		items = append(items, gin.H{
			"id": cv.ID, "title": cv.Title, "message_count": cv.MessageCount,
			"updated_at":  cv.UpdatedAt.Format(time.RFC3339),
			"created_at":  cv.CreatedAt.Format(time.RFC3339),
			"has_summary": cv.Summary != "",
		})
	}
	c.JSON(http.StatusOK, gin.H{"conversations": items})
}

func (a *API) CreateConversation(c *gin.Context) {
	var req struct {
		Title string `json:"title"`
	}
	_ = c.ShouldBindJSON(&req)
	id := newUUID()
	if err := a.st.CreateConversation(c.Request.Context(), id, userIDFrom(c), strings.TrimSpace(req.Title)); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"id": id, "title": orDefault(req.Title, "新会话")})
}

func (a *API) GetMessages(c *gin.Context) {
	convID := c.Param("id")
	conv, err := a.st.GetConversation(c.Request.Context(), convID)
	if err != nil || conv.UserID != userIDFrom(c) {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}
	msgs, err := a.st.ListMessages(c.Request.Context(), convID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	items := make([]gin.H, 0, len(msgs))
	for _, m := range msgs {
		meta := map[string]any{}
		if m.Meta != "" {
			_ = json.Unmarshal([]byte(m.Meta), &meta)
		}
		var toolCalls []schemaToolCallLite
		if m.ToolCalls != "" {
			_ = json.Unmarshal([]byte(m.ToolCalls), &toolCalls)
		}
		items = append(items, gin.H{
			"id": m.ID, "seq": m.Seq, "role": m.Role,
			"content": m.Content, "reasoning": m.Reasoning,
			"tool_name": m.ToolName, "tool_call_id": m.ToolCallID,
			"tool_calls": toolCalls,
			"meta":       meta,
			"created_at": m.CreatedAt.Format(time.RFC3339),
		})
	}
	c.JSON(http.StatusOK, gin.H{
		"conversation": gin.H{"id": conv.ID, "title": conv.Title, "has_summary": conv.Summary != ""},
		"messages":     items,
	})
}

type schemaToolCallLite struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (a *API) RenameConversation(c *gin.Context) {
	var req struct {
		Title string `json:"title"`
	}
	if err := c.ShouldBindJSON(&req); err != nil || strings.TrimSpace(req.Title) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "标题不能为空"})
		return
	}
	if err := a.st.RenameConversation(c.Request.Context(), c.Param("id"), userIDFrom(c), strings.TrimSpace(req.Title)); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"ok": true})
}

func (a *API) DeleteConversation(c *gin.Context) {
	err := a.st.DeleteConversation(c.Request.Context(), c.Param("id"), userIDFrom(c))
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"ok": true})
}

// ---------- chat ----------

type chatRequest struct {
	ConversationID string `json:"conversation_id"`
	Message        string `json:"message"`
}

func (a *API) prepareConversation(c *gin.Context, convID string) (*store.Conversation, error) {
	uid := userIDFrom(c)
	if strings.TrimSpace(convID) == "" {
		id := newUUID()
		if err := a.st.CreateConversation(c.Request.Context(), id, uid, "新会话"); err != nil {
			return nil, err
		}
		return &store.Conversation{ID: id, UserID: uid, Title: "新会话"}, nil
	}
	conv, err := a.st.GetConversation(c.Request.Context(), convID)
	if err != nil {
		return nil, err
	}
	if conv.UserID != uid {
		return nil, store.ErrNotFound
	}
	return conv, nil
}

func (a *API) Chat(c *gin.Context) {
	var req chatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误"})
		return
	}
	req.Message = strings.TrimSpace(req.Message)
	if req.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "消息不能为空"})
		return
	}
	if len([]rune(req.Message)) > 4000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "消息过长，请控制在 4000 字以内"})
		return
	}
	conv, err := a.prepareConversation(c, req.ConversationID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	res, err := a.agent.Run(c.Request.Context(), userIDFrom(c), conv.ID, req.Message, false, nil)
	if err != nil {
		log.Printf("非流式对话失败: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{"error": "模型调用失败：" + err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"conversation_id": conv.ID,
		"title":           res.Title,
		"content":         res.Content,
		"reasoning":       res.Reasoning,
		"steps":           res.Steps,
		"usage":           res.Usage,
	})
}

func (a *API) ChatStream(c *gin.Context) {
	var req chatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请求格式错误"})
		return
	}
	req.Message = strings.TrimSpace(req.Message)
	if req.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "消息不能为空"})
		return
	}
	if len([]rune(req.Message)) > 4000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "消息过长，请控制在 4000 字以内"})
		return
	}
	conv, err := a.prepareConversation(c, req.ConversationID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "会话不存在"})
		return
	}

	c.Header("Content-Type", "text/event-stream; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "当前服务器不支持流式输出"})
		return
	}

	var sentErr bool
	send := func(ev agent.Event) {
		if ev.Type == agent.EventError {
			sentErr = true
		}
		data, err := json.Marshal(ev)
		if err != nil {
			return
		}
		fmt.Fprintf(c.Writer, "event: %s\ndata: %s\n\n", ev.Type, data)
		flusher.Flush()
	}

	// 先发一条 meta，让前端立刻拿到会话 ID（新建会话时可立即加入侧边栏）
	send(agent.Event{Type: "meta", ConversationID: conv.ID, Text: orDefault(conv.Title, "新会话")})

	res, err := a.agent.Run(c.Request.Context(), userIDFrom(c), conv.ID, req.Message, true, send)
	if err != nil {
		log.Printf("流式对话失败: %v", err)
		if !sentErr {
			send(agent.Event{Type: agent.EventError, ConversationID: conv.ID, Text: "对话失败：" + err.Error()})
		}
		return
	}
	// 标题可能在本轮首次生成，补发一条 meta 让前端立即更新
	if res != nil && res.Title != "" {
		send(agent.Event{Type: "meta", ConversationID: conv.ID, Text: res.Title})
	}
}

// ---------- documents ----------

func (a *API) UploadDocument(c *gin.Context) {
	fh, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请选择要上传的文件"})
		return
	}
	if fh.Size > 10<<20 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "文件不能超过 10MB"})
		return
	}
	ext := strings.ToLower(fh.Filename[strings.LastIndex(fh.Filename, ".")+1:])
	switch ext {
	case "txt", "md", "markdown", "docx":
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "仅支持 .txt / .md / .docx"})
		return
	}

	file, err := fh.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "读取文件失败"})
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 10<<20))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "读取文件失败"})
		return
	}

	uid := userIDFrom(c)
	docID := newUUID()
	doc := store.Document{
		ID: docID, UserID: uid, Name: fh.Filename, Size: fh.Size,
		Status: "processing", ChunkCount: 0,
	}
	if err := a.st.CreateDocument(c.Request.Context(), doc); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "创建文档记录失败"})
		return
	}

	chunks, err := a.rag.Ingest(c.Request.Context(), uid, docID, fh.Filename, fh.Size, data)
	if err != nil {
		_ = a.st.UpdateDocumentStatus(c.Request.Context(), docID, "failed", err.Error())
		c.JSON(http.StatusBadRequest, gin.H{"error": "文档处理失败：" + err.Error()})
		return
	}
	_ = a.st.UpdateDocumentChunkCount(c.Request.Context(), docID, chunks)
	_ = a.st.UpdateDocumentStatus(c.Request.Context(), docID, "ready", "")

	c.JSON(http.StatusOK, gin.H{
		"id": docID, "name": fh.Filename, "size": fh.Size,
		"chunk_count": chunks, "status": "ready",
	})
}

func (a *API) ListDocuments(c *gin.Context) {
	docs, err := a.st.ListDocuments(c.Request.Context(), userIDFrom(c))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	items := make([]gin.H, 0, len(docs))
	for _, d := range docs {
		items = append(items, gin.H{
			"id": d.ID, "name": d.Name, "size": d.Size, "chunk_count": d.ChunkCount,
			"status": d.Status, "error": d.ErrMsg, "created_at": d.CreatedAt.Format(time.RFC3339),
		})
	}
	c.JSON(http.StatusOK, gin.H{"documents": items})
}

func (a *API) DeleteDocument(c *gin.Context) {
	uid := userIDFrom(c)
	if err := a.st.DeleteDocument(c.Request.Context(), c.Param("id"), uid); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "文档不存在"})
		return
	}
	a.rag.RemoveDocument(c.Request.Context(), c.Param("id"), uid)
	c.JSON(http.StatusOK, gin.H{"ok": true})
}

// ---------- misc ----------

func (a *API) Stats(c *gin.Context) {
	uid := userIDFrom(c)
	docs, _ := a.st.ListDocuments(c.Request.Context(), uid)
	chunks, _ := a.st.CountUserChunks(c.Request.Context(), uid)
	convs, _ := a.st.ListConversations(c.Request.Context(), uid)
	c.JSON(http.StatusOK, gin.H{
		"model":         a.cfg.LLMModel,
		"base_url":      a.cfg.LLMBaseURL,
		"embedder":      a.rag.EmbedderName(),
		"vector_store":  a.rag.VectorStoreName(),
		"documents":     len(docs),
		"chunks":        chunks,
		"conversations": len(convs),
		"redis":         a.cacheEnabled,
	})
}

func (a *API) Health(c *gin.Context) {
	err := a.st.DB.PingContext(c.Request.Context())
	c.JSON(http.StatusOK, gin.H{"status": "ok", "mysql": err == nil})
}

func (a *API) NotFound(c *gin.Context) {
	c.JSON(http.StatusNotFound, gin.H{"error": "not found", "path": c.Request.URL.Path})
}
