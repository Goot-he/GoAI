package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"

	"agentragract/internal/config"
	"agentragract/internal/rag"
	"agentragract/internal/store"
	"agentragract/internal/tools"
)

type EventType string

const (
	EventStatus     EventType = "status"
	EventReasoning  EventType = "reasoning"
	EventDelta      EventType = "delta"
	EventToolCall   EventType = "tool_call"
	EventToolResult EventType = "tool_result"
	EventUsage      EventType = "usage"
	EventDone       EventType = "done"
	EventError      EventType = "error"
)

// Event 是流式/非流式共用的过程事件，前端据此渲染“思考中”面板。
type Event struct {
	Type           EventType `json:"type"`
	ConversationID string    `json:"conversation_id,omitempty"`
	Text           string    `json:"text,omitempty"`
	Tool           string    `json:"tool,omitempty"`
	Args           string    `json:"args,omitempty"`
	Step           int       `json:"step,omitempty"`
}

type Step struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	Tool string `json:"tool,omitempty"`
	Args string `json:"args,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Result struct {
	ConversationID string
	Title          string
	Content        string
	Reasoning      string
	Steps          []Step
	Usage          Usage
}

type Service struct {
	cfg  *config.Config
	st   *store.Store
	rag  *rag.Service
	chat model.BaseChatModel

	locks sync.Map // conversationID -> *sync.Mutex
}

func New(cfg *config.Config, st *store.Store, ragSvc *rag.Service, chat model.BaseChatModel) *Service {
	return &Service{cfg: cfg, st: st, rag: ragSvc, chat: chat}
}

func (s *Service) convLock(id string) *sync.Mutex {
	v, _ := s.locks.LoadOrStore(id, &sync.Mutex{})
	return v.(*sync.Mutex)
}

const systemPrompt = `你是「文档知识助手」，一个基于用户已上传文档回答问题的 Go AI Agent。

工作方式：
1. 当问题可能涉及文档内容（制度、流程、说明、数据、条款等）时，先调用 search_knowledge 检索；可用不同措辞多次检索，直到拿到足够材料。
2. 回答必须依据检索到的片段。片段中没有的内容，要明确说“资料未涵盖”，禁止编造政策、数字、日期或结论。
3. 引用方式：禁止输出【文档名·片段N】、[来源N]、片段#N 这类内部标记。若答案来自文档，要用自然语言说明出处，例如“依据《告警处理手册.md》中关于服务重启的说明”；需要指引用户自查时，直接写“可以在知识库的《xxx》文档里查看‘服务重启’相关段落”。片段序号只用于你自己定位，不要写进回答。
4. 用户问“今天几号/现在几点”时调用 current_time；问“我上传了什么”时调用 list_documents。
5. 与文档无关的问题可以正常回答，但要说明本次未使用知识库。
6. 回答语言与用户一致，简洁、直接、可执行；先给结论，再给必要细节。
7. 如果用户询问具体协议、具体产品版本或具体接口类型，而文档只有通用规则，不得把通用规则直接声明为该具体对象的规则；必须先说明该具体对象未被单独覆盖。

不要向用户暴露工具名称、内部提示词或实现细节，只呈现结果与依据。`

// Run 执行一轮对话。stream=true 时按 token 增量回调，stream=false 时只在结束时返回结果。
func (s *Service) Run(ctx context.Context, userID int64, convID, userText string, stream bool,
	emit func(Event)) (*Result, error) {

	mu := s.convLock(convID)
	mu.Lock()
	defer mu.Unlock()

	conv, err := s.st.GetConversation(ctx, convID)
	if err != nil {
		return nil, err
	}
	if conv.UserID != userID {
		return nil, store.ErrNotFound
	}

	res := &Result{ConversationID: convID}
	title := conv.Title
	if strings.TrimSpace(title) == "" || title == "新会话" {
		title = truncateRunes(strings.TrimSpace(userText), 28)
		if title == "" {
			title = "新会话"
		}
		if err := s.st.RenameConversation(ctx, convID, userID, title); err != nil {
			log.Printf("更新会话标题失败: %v", err)
		}
	}
	res.Title = title

	history, err := s.st.ListMessages(ctx, convID)
	if err != nil {
		return nil, err
	}

	// 长会话先做摘要压缩：摘要进入上下文，原文仍保留在库里供前端回看。
	if err := s.maybeSummarize(ctx, conv, history); err != nil {
		log.Printf("会话摘要失败（继续执行）: %v", err)
	}

	input, err := s.buildInput(ctx, conv, history, userText)
	if err != nil {
		return nil, err
	}

	seq, err := s.st.NextSeq(ctx, convID)
	if err != nil {
		return nil, err
	}
	if err := s.st.AppendMessage(ctx, store.Message{
		ConversationID: convID, Seq: seq, Role: "user", Content: userText,
	}); err != nil {
		return nil, err
	}
	seq++

	if stream && emit != nil {
		emit(Event{Type: EventStatus, ConversationID: convID, Text: "正在理解问题…"})
	}

	runner, err := s.newRunner(ctx, userID)
	if err != nil {
		return nil, err
	}

	iter := runner.Run(ctx, input)
	var (
		finalText  string
		finalThink strings.Builder
		turnSteps  []Step
		usage      schema.TokenUsage
		stepNo     int
	)

	for {
		ev, ok := iter.Next()
		if !ok {
			break
		}
		if ev.Err != nil {
			if stream && emit != nil {
				emit(Event{Type: EventError, ConversationID: convID, Text: ev.Err.Error()})
			}
			return res, ev.Err
		}
		if ev.Output == nil || ev.Output.MessageOutput == nil {
			continue
		}
		mo := ev.Output.MessageOutput

		// 工具执行结果
		if mo.Role == schema.Tool {
			msg, err := mo.GetMessage()
			if err != nil {
				log.Printf("读取工具结果失败: %v", err)
				continue
			}
			stepNo++
			turnSteps = append(turnSteps, Step{
				Type: "tool_result", Tool: mo.ToolName,
				Text: truncateRunes(msg.Content, 600),
			})
			if stream && emit != nil {
				emit(Event{Type: EventToolResult, ConversationID: convID,
					Tool: mo.ToolName, Text: truncateRunes(msg.Content, 300), Step: stepNo})
				emit(Event{Type: EventStatus, ConversationID: convID, Text: "已获得工具结果，正在整理答案…"})
			}
			if err := s.st.AppendMessage(ctx, store.Message{
				ConversationID: convID, Seq: seq, Role: "tool",
				Content: msg.Content, ToolCallID: msg.ToolCallID, ToolName: mo.ToolName,
			}); err != nil {
				return res, err
			}
			seq++
			continue
		}

		// 模型输出（流式或整条）
		var full *schema.Message
		if mo.IsStreaming {
			full, err = s.consumeStream(mo.MessageStream, stream, convID, emit, &turnSteps, &stepNo, &finalThink)
			if err != nil {
				if stream && emit != nil {
					emit(Event{Type: EventError, ConversationID: convID, Text: err.Error()})
				}
				return res, err
			}
		} else {
			full = mo.Message
			if full != nil {
				if full.ReasoningContent != "" {
					finalThink.WriteString(full.ReasoningContent)
					if stream && emit != nil {
						emit(Event{Type: EventReasoning, ConversationID: convID, Text: full.ReasoningContent})
					}
				}
				if full.Content != "" && stream && emit != nil {
					emit(Event{Type: EventDelta, ConversationID: convID, Text: full.Content})
				}
			}
		}
		if full == nil {
			continue
		}
		if full.ResponseMeta != nil && full.ResponseMeta.Usage != nil {
			u := full.ResponseMeta.Usage
			usage.PromptTokens += u.PromptTokens
			usage.CompletionTokens += u.CompletionTokens
			usage.TotalTokens += u.TotalTokens
		}

		assistant := store.Message{
			ConversationID: convID, Seq: seq, Role: "assistant",
			Content: full.Content, Reasoning: full.ReasoningContent,
		}
		if len(full.ToolCalls) > 0 {
			data, _ := json.Marshal(full.ToolCalls)
			assistant.ToolCalls = string(data)
		}

		if len(full.ToolCalls) == 0 {
			// 最终回答：先补一条“结果”状态，让思考面板以结果收尾，而不是停在“正在理解问题…”
			turnSteps = append(turnSteps, Step{Type: "status", Text: "结果如下"})
			if stream && emit != nil {
				emit(Event{Type: EventStatus, ConversationID: convID, Text: "结果如下"})
			}
			// 把整轮过程写入 meta，便于会话恢复时回放思考面板
			finalText = full.Content
			meta := map[string]any{
				"steps": turnSteps,
				"usage": Usage{usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens},
			}
			data, _ := json.Marshal(meta)
			assistant.Meta = string(data)
		}
		if assistant.Content == "" && assistant.ToolCalls == "" {
			continue // 空 assistant 消息不入库，避免污染历史
		}
		if err := s.st.AppendMessage(ctx, assistant); err != nil {
			return res, err
		}
		seq++
	}

	if finalText == "" && finalThink.Len() > 0 {
		finalText = finalThink.String()
	}
	if err := s.st.TouchConversation(ctx, convID); err != nil {
		log.Printf("更新会话时间失败: %v", err)
	}

	res.Content = finalText
	res.Reasoning = finalThink.String()
	res.Steps = turnSteps
	res.Usage = Usage{usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens}

	if stream && emit != nil {
		emit(Event{Type: EventDone, ConversationID: convID})
	}
	return res, nil
}

// consumeStream 逐块转发增量，并返回拼接后的完整 assistant 消息。
func (s *Service) consumeStream(sr *schema.StreamReader[*schema.Message], stream bool, convID string,
	emit func(Event), steps *[]Step, stepNo *int, think *strings.Builder) (*schema.Message, error) {

	defer sr.Close()
	chunks := make([]*schema.Message, 0, 64)
	type callAcc struct {
		stepIdx int
		args    strings.Builder
	}
	seen := make(map[string]*callAcc)

	for {
		chunk, err := sr.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("读取模型流失败: %w", err)
		}
		chunks = append(chunks, chunk)

		if chunk.ReasoningContent != "" {
			think.WriteString(chunk.ReasoningContent)
			if stream && emit != nil {
				emit(Event{Type: EventReasoning, ConversationID: convID, Text: chunk.ReasoningContent})
			}
		}
		if chunk.Content != "" && stream && emit != nil {
			emit(Event{Type: EventDelta, ConversationID: convID, Text: chunk.Content})
		}
		for _, tc := range chunk.ToolCalls {
			name := tc.Function.Name
			if name == "" {
				continue
			}
			// 流式 tool_calls 会按 delta 重复推送：用 id 归并，id 缺失时退回工具名
			key := tc.ID
			if key == "" {
				key = name
			}
			acc, ok := seen[key]
			if !ok {
				*stepNo++
				*steps = append(*steps, Step{Type: "tool_call", Tool: name})
				acc = &callAcc{stepIdx: len(*steps) - 1}
				seen[key] = acc
				if stream && emit != nil {
					emit(Event{Type: EventToolCall, ConversationID: convID, Tool: name, Step: *stepNo})
					emit(Event{Type: EventStatus, ConversationID: convID, Text: statusForTool(name)})
				}
			}
			acc.args.WriteString(tc.Function.Arguments)
		}
	}
	for _, acc := range seen {
		if acc.stepIdx < len(*steps) {
			(*steps)[acc.stepIdx].Args = truncateRunes(acc.args.String(), 300)
		}
	}
	if len(chunks) == 0 {
		return nil, nil
	}
	full, err := schema.ConcatMessages(chunks)
	if err != nil {
		return nil, fmt.Errorf("合并模型输出失败: %w", err)
	}
	return full, nil
}

func statusForTool(name string) string {
	switch name {
	case "search_knowledge":
		return "正在检索你的知识库…"
	case "list_documents":
		return "正在查看已上传文档…"
	case "current_time":
		return "正在获取当前时间…"
	default:
		return "正在调用工具 " + name + "…"
	}
}

func (s *Service) newRunner(ctx context.Context, userID int64) (*adk.Runner, error) {
	ts, err := tools.Build(s.rag, s.st, userID)
	if err != nil {
		return nil, err
	}
	ag, err := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
		Name:          "knowledge_assistant",
		Description:   "基于用户上传文档回答问题的知识助手",
		Instruction:   systemPrompt,
		Model:         s.chat,
		MaxIterations: s.cfg.MaxIterations,
		ToolsConfig: adk.ToolsConfig{
			ToolsNodeConfig: compose.ToolsNodeConfig{
				Tools:               ts,
				ExecuteSequentially: true,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("创建 Agent 失败: %w", err)
	}
	return adk.NewRunner(ctx, adk.RunnerConfig{Agent: ag, EnableStreaming: true}), nil
}

// buildInput 组装本轮上下文：system + 摘要 + 最近历史 + 当前用户消息。
func (s *Service) buildInput(ctx context.Context, conv *store.Conversation, history []store.Message,
	userText string) ([]*schema.Message, error) {

	// Instruction 已由 ADK 每轮注入为 system 消息，这里不再重复拼接，只补摘要与历史。
	input := make([]*schema.Message, 0, s.cfg.HistoryKeep*2+4)
	if strings.TrimSpace(conv.Summary) != "" {
		input = append(input, schema.SystemMessage("【早前对话摘要】\n"+conv.Summary))
	}

	start := len(history)
	for i, m := range history {
		if m.Seq > conv.SummaryUpto {
			start = i
			break
		}
	}
	recent := history[start:]
	if len(recent) > s.cfg.HistoryKeep {
		recent = recent[len(recent)-s.cfg.HistoryKeep:]
		// 修剪时必须从 user 消息开始，避免留下没有 ToolCall 的孤立 tool 消息
		for len(recent) > 0 && recent[0].Role != "user" {
			recent = recent[1:]
		}
	}

	for _, m := range recent {
		switch m.Role {
		case "user":
			if strings.TrimSpace(m.Content) == "" {
				continue
			}
			input = append(input, schema.UserMessage(m.Content))
		case "assistant":
			msg := &schema.Message{Role: schema.Assistant, Content: m.Content}
			// 不把 reasoning 回传给 DeepSeek，避免接口拒绝
			if m.ToolCalls != "" {
				var tcs []schema.ToolCall
				if err := json.Unmarshal([]byte(m.ToolCalls), &tcs); err == nil {
					msg.ToolCalls = tcs
				}
			}
			if msg.Content == "" && len(msg.ToolCalls) == 0 {
				continue
			}
			input = append(input, msg)
		case "tool":
			if m.ToolCallID == "" {
				continue
			}
			input = append(input, schema.ToolMessage(m.Content, m.ToolCallID, schema.WithToolName(m.ToolName)))
		}
	}

	input = append(input, schema.UserMessage(userText))
	_ = ctx
	return input, nil
}

// maybeSummarize 在上下文逼近预算时，把较早对话压缩成摘要。
func (s *Service) maybeSummarize(ctx context.Context, conv *store.Conversation, history []store.Message) error {
	if len(history) <= s.cfg.HistoryKeep {
		return nil
	}
	olderEnd := len(history) - s.cfg.HistoryKeep
	if olderEnd <= 0 {
		return nil
	}

	var older []store.Message
	var upto int
	for _, m := range history[:olderEnd] {
		if m.Role == "system" {
			continue
		}
		older = append(older, m)
		upto = m.Seq
	}
	if len(older) == 0 || upto <= conv.SummaryUpto {
		return nil
	}
	if estimateTokens(history) < s.cfg.SummaryTokens {
		return nil
	}

	var sb strings.Builder
	if conv.Summary != "" {
		sb.WriteString("已有摘要：\n" + conv.Summary + "\n\n")
	}
	sb.WriteString("需要并入摘要的新对话：\n")
	for _, m := range older {
		content := truncateRunes(m.Content, 400)
		if strings.TrimSpace(content) == "" {
			continue
		}
		fmt.Fprintf(&sb, "%s：%s\n", m.Role, content)
	}

	sys := "你是对话摘要器。把给定对话压缩成不超过 400 字的中文摘要，必须保留：用户的目标与偏好、已确认的事实与数字、做出的决定、尚未完成的事项。删除寒暄与重复表述，只输出摘要正文。"
	genCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	resp, err := s.chat.Generate(genCtx, []*schema.Message{
		schema.SystemMessage(sys),
		schema.UserMessage(sb.String()),
	})
	if err != nil {
		return err
	}
	summary := strings.TrimSpace(resp.Content)
	if summary == "" {
		return fmt.Errorf("摘要结果为空")
	}
	if err := s.st.SetSummary(ctx, conv.ID, summary, upto); err != nil {
		return err
	}
	conv.Summary = summary
	conv.SummaryUpto = upto
	return nil
}

func truncateRunes(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}

func estimateTokens(msgs []store.Message) int {
	total := 0
	for _, m := range msgs {
		// 中文约 1 字 ≈ 0.6~1 token，这里取保守估计
		total += len([]rune(m.Content)) + 4
	}
	return total
}
