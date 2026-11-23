package tools

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/components/tool/utils"

	"agentragract/internal/rag"
	"agentragract/internal/store"
)

type SearchArgs struct {
	Query string `json:"query" jsonschema:"required" jsonschema_description:"检索用的自然语言问题或关键词，尽量具体，例如“退货期限是几天”"`
	TopK  int    `json:"top_k,omitempty" jsonschema_description:"返回片段数量，1-8，默认 5"`
}

type DocumentArgs struct{}

type TimeArgs struct {
	Timezone string `json:"timezone,omitempty" jsonschema_description:"IANA 时区名，默认 Asia/Shanghai"`
}

// Build 为指定用户构造工具集。用户身份通过闭包绑定，
// 模型生成的参数无法越权访问其他用户的知识库（对应课程 M11 7 的身份绑定原则）。
func Build(ragSvc *rag.Service, st *store.Store, userID int64) ([]tool.BaseTool, error) {
	searchTool, err := utils.InferTool[SearchArgs, string](
		"search_knowledge",
		"在当前用户上传的文档知识库中检索资料。涉及文档内容、制度、说明、数据等问题时，先调用本工具；可以用不同措辞多次检索。返回带来源的片段。",
		func(ctx context.Context, args SearchArgs) (string, error) {
			if strings.TrimSpace(args.Query) == "" {
				return "", fmt.Errorf("query 不能为空")
			}
			topK := args.TopK
			if topK <= 0 {
				topK = 5
			}
			if topK > 8 {
				topK = 8
			}
			hits, err := ragSvc.Search(ctx, userID, args.Query, topK)
			if err != nil {
				return "", fmt.Errorf("知识库检索失败: %w", err)
			}
			return rag.FormatHits(hits), nil
		},
	)
	if err != nil {
		return nil, err
	}

	listTool, err := utils.InferTool[DocumentArgs, string](
		"list_documents",
		"列出当前用户已经上传到知识库的文档名称与片段数量。用户询问“我上传了什么/有哪些资料”时使用。",
		func(ctx context.Context, _ DocumentArgs) (string, error) {
			docs, err := st.ListDocuments(ctx, userID)
			if err != nil {
				return "", err
			}
			if len(docs) == 0 {
				return "当前用户还没有上传任何文档。", nil
			}
			var sb strings.Builder
			for _, d := range docs {
				fmt.Fprintf(&sb, "- 《%s》（%d 个片段，状态 %s）\n", d.Name, d.ChunkCount, d.Status)
			}
			return sb.String(), nil
		},
	)
	if err != nil {
		return nil, err
	}

	timeTool, err := utils.InferTool[TimeArgs, string](
		"current_time",
		"获取当前日期与时间。用户问“今天几号/现在几点”等与时间相关的问题时使用。",
		func(_ context.Context, args TimeArgs) (string, error) {
			tz := args.Timezone
			if tz == "" {
				tz = "Asia/Shanghai"
			}
			loc, err := time.LoadLocation(tz)
			if err != nil {
				return "", fmt.Errorf("未知时区 %q", tz)
			}
			return time.Now().In(loc).Format("2006-01-02 15:04:05 MST"), nil
		},
	)
	if err != nil {
		return nil, err
	}

	return []tool.BaseTool{searchTool, listTool, timeTool}, nil
}
