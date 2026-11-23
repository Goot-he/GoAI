# Agent-Rag-Ract

一个基于 **Go + Gin + Eino + DeepSeek** 的文档知识助手（RAG Agent），分层设计遵循 `skills/go-ai-agent-course` 教程提炼的落地路径：

```
Provider → Prompt → Agent 循环 → 工具 → RAG → 记忆 → 上下文治理 → 可观测
```

## 功能一览

- **流式 / 非流式可选**：SSE 逐字输出，或一次性返回；前端一个开关切换。
- **思考过程可视化**：状态时间线（理解问题 → 检索知识库 → 整理答案）、模型推理内容、工具调用与工具返回结果，结束时全部可回放。
- **文档 RAG**：上传 `.txt / .md / .docx`，自动切分 → Ollama `nomic-embed-text` 向量化（768 维）→ 写入 Qdrant；回答时先检索，再用自然语言说明出自哪份文档。
- **混合检索**：Qdrant 向量召回（HNSW + 余弦，按 user_id 过滤）+ BM25 关键词排名，两路分值各自归一化后加权融合（0.45 / 0.55），向量库不可用时自动降级为纯关键词检索。
- **完整记忆**：同一会话按 seq 全量落库，每轮带最近 12 条历史；超长会话自动摘要压缩（保留摘要 + 原文回看）。
- **会话恢复**：登录后侧边栏即历史会话，点开恢复全部消息、思考面板与工具轨迹。
- **Gin + Eino**：Eino ADK `ChatModelAgent` + `Runner` 驱动工具调用循环，Gin 提供 REST + SSE。
- **MySQL + Qdrant + Redis**：消息、文档、分片正文持久化在 MySQL；向量存 Qdrant（point id = chunks.id），启动时自动回填历史数据；Redis 只做查询向量与检索结果缓存，不可用时进程内降级。

## 快速开始
docker-compose.yml：这个文件是开发者在本地打包好的一些环境依赖，可以通过这个文件在任何一台电脑上启动这个项目。
```bash
# 1. 起依赖（MySQL 8 + Redis 7 + Qdrant）
# 若本机已有 MySQL 占用 3306，compose 已把容器映射到 3307
docker compose up -d

# 2. 准备向量模型（本机 Ollama）
ollama pull nomic-embed-text

# 2. 配置：把 .env.example 复制为 .env，填入 LLM_API_KEY / JWT_SECRET
#    （.env 已被 .gitignore 忽略）

# 3. 启动（PowerShell，会自动加载 .env）
.\run.ps1
```

打开 http://localhost:8080 ，注册 → 上传文档 → 提问。

也可以不写文件，直接用环境变量：`$env:LLM_API_KEY="sk-..."; go run ./cmd/server`。全部配置项见 `.env.example`。

## 目录结构

```
cmd/server/            启动装配：config → store → cache → embedder → rag → model → agent → router
internal/config/       环境变量配置与启动校验
internal/store/        MySQL 连接、自动建库建表、User/Conversation/Message/Document/Chunk CRUD
internal/cache/        Redis 缓存，失败自动降级为带 TTL 的进程内 map
internal/rag/rag.go    文本提取、切分、Embedder（Hash / OpenAI 兼容，接 Ollama nomic-embed-text）、向量编解码、余弦相似度
internal/rag/qdrant.go Qdrant REST 客户端：集合创建、payload 索引、Upsert、ANN 检索、按文档删除、点数统计
internal/rag/index.go  用户级 BM25 索引、Qdrant 向量召回、归一化加权融合、检索缓存、历史数据回填
internal/tools/        3 个 Eino 工具：search_knowledge / list_documents / current_time
internal/agent/        Agent 循环、事件流、上下文组装、摘要压缩、消息落库
internal/httpapi/      JWT 鉴权、会话/文档/聊天/SSE 接口、内嵌前端路由
internal/webui/        index.html + app.css + app.js（//go:embed，无构建步骤）
```

## 关键接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/register` `/api/auth/login` | 邮箱密码注册 / 登录，返回 JWT |
| GET | `/api/auth/me` | 当前用户 |
| GET/POST | `/api/conversations` | 会话列表 / 新建 |
| GET | `/api/conversations/:id/messages` | 恢复历史（含 meta.steps 思考轨迹） |
| PATCH/DELETE | `/api/conversations/:id` | 重命名 / 删除 |
| POST | `/api/chat` | 非流式对话 |
| POST | `/api/chat/stream` | SSE 流式对话 |
| POST/GET/DELETE | `/api/documents` | 上传（multipart `file`）/ 列表 / 删除 |
| GET | `/api/stats` | 模型、向量器、文档/片段/会话统计 |

SSE 事件类型：`meta`、`status`、`reasoning`、`delta`、`tool_call`、`tool_result`、`usage`、`done`、`error`。

## 设计要点

**Agent 循环**：`adk.NewChatModelAgent` 挂 3 个工具，`MaxIterations` 限制循环次数，`Runner` 以事件流方式把模型增量、工具结果逐条抛出，`internal/agent` 负责转成前端事件并同步落库。

**上下文**：system prompt（ADK Instruction）+ 早前摘要 + 最近 `HISTORY_KEEP` 条消息。裁剪时强制从 `user` 消息开始，避免留下孤立的 `tool` 消息导致上游 400；历史里的 `reasoning_content` 不回传（DeepSeek 会拒绝）。

**记忆不丢**：消息带自增 `seq` 与唯一索引，历史按 seq 回放；只有无 tool_calls 的最终回答才把本轮 steps/usage 写进 meta。摘要只影响“喂给模型”的窗口，原文永不被删。

**工具安全**：每个工具用闭包绑定当前 userID，检索与文档列表只能命中自己的数据，天然隔离多用户。

## 注意

- **模型名**：DeepSeek 官方只有 `deepseek-chat` / `deepseek-reasoner`。如果使用的是提供 “flash” 命名的兼容网关，改 `LLM_MODEL` 与 `LLM_BASE_URL` 即可，代码不做硬编码。带思考过程的模型请设 `deepseek-reasoner`。
- **向量维度**：默认 Ollama `nomic-embed-text` = 768 维，与 `EMBEDDING_DIM`、Qdrant 集合维度必须一致。更换 embedding 模型后新旧向量不在同一空间，需清空集合与知识库后重新上传；启动时若集合为空会自动把 MySQL 里的历史片段重嵌入回填。
- **降级策略**：`QDRANT_URL` 留空或 Qdrant 初始化失败时，退化为纯 BM25 关键词检索；Ollama 查询向量化失败时当次请求同样降级，不影响对话主链路。
- **超时**：SSE 是长连接，服务端未设 `WriteTimeout`，整体受 `LLM_TIMEOUT_SECONDS` 与请求 context 约束。
