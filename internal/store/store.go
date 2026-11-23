package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/go-sql-driver/mysql"

	"agentragract/internal/config"
)

var (
	ErrNotFound = errors.New("not found")
	ErrConflict = errors.New("conflict")
)

type Store struct {
	DB *sql.DB
}

func New(ctx context.Context, cfg *config.Config) (*Store, error) {
	if err := ensureDatabase(ctx, cfg.MySQLDSN); err != nil {
		return nil, err
	}

	db, err := sql.Open("mysql", cfg.MySQLDSN)
	if err != nil {
		return nil, fmt.Errorf("打开 MySQL 连接失败: %w", err)
	}
	db.SetMaxOpenConns(20)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(time.Hour)
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("连接 MySQL 失败: %w", err)
	}

	s := &Store{DB: db}
	if err := s.migrate(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// ensureDatabase 先连到 server 层创建库，避免手工初始化。
func ensureDatabase(ctx context.Context, dsn string) error {
	conf, err := mysql.ParseDSN(dsn)
	if err != nil {
		return fmt.Errorf("MySQL_DSN 解析失败: %w", err)
	}
	name := conf.DBName
	if name == "" {
		return fmt.Errorf("MySQL_DSN 必须包含数据库名")
	}
	conf.DBName = ""
	serverDSN := conf.FormatDSN()
	db, err := sql.Open("mysql", serverDSN)
	if err != nil {
		return fmt.Errorf("连接 MySQL Server 失败: %w", err)
	}
	defer db.Close()
	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("连接 MySQL Server 失败: %w", err)
	}
	_, err = db.ExecContext(ctx, "CREATE DATABASE IF NOT EXISTS `"+name+"` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
	if err != nil {
		return fmt.Errorf("创建数据库失败: %w", err)
	}
	return nil
}

func (s *Store) migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			email VARCHAR(190) NOT NULL UNIQUE,
			password_hash VARCHAR(100) NOT NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,

		`CREATE TABLE IF NOT EXISTS conversations (
			id VARCHAR(36) PRIMARY KEY,
			user_id BIGINT NOT NULL,
			title VARCHAR(200) NOT NULL DEFAULT '',
			summary LONGTEXT NULL,
			summary_upto INT NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			KEY idx_conv_user (user_id, updated_at)
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,

		`CREATE TABLE IF NOT EXISTS messages (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			conversation_id VARCHAR(36) NOT NULL,
			seq INT NOT NULL,
			role VARCHAR(20) NOT NULL,
			content LONGTEXT NULL,
			reasoning LONGTEXT NULL,
			tool_calls LONGTEXT NULL,
			tool_call_id VARCHAR(80) NULL,
			tool_name VARCHAR(80) NULL,
			meta LONGTEXT NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			UNIQUE KEY uk_conv_seq (conversation_id, seq),
			KEY idx_conv (conversation_id)
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,

		`CREATE TABLE IF NOT EXISTS documents (
			id VARCHAR(36) PRIMARY KEY,
			user_id BIGINT NOT NULL,
			name VARCHAR(255) NOT NULL,
			size BIGINT NOT NULL DEFAULT 0,
			chunk_count INT NOT NULL DEFAULT 0,
			status VARCHAR(20) NOT NULL DEFAULT 'ready',
			err_msg VARCHAR(255) NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			KEY idx_doc_user (user_id, created_at)
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,

		`CREATE TABLE IF NOT EXISTS chunks (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			document_id VARCHAR(36) NOT NULL,
			user_id BIGINT NOT NULL,
			seq INT NOT NULL,
			doc_name VARCHAR(255) NOT NULL,
			content LONGTEXT NOT NULL,
			embedding MEDIUMBLOB NULL,
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			KEY idx_chunk_user (user_id),
			KEY idx_chunk_doc (document_id)
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`,
	}
	for _, stmt := range stmts {
		if _, err := s.DB.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("执行建表语句失败: %w", err)
		}
	}
	return nil
}

func (s *Store) Close() error { return s.DB.Close() }

// ---------- users ----------

type User struct {
	ID        int64
	Email     string
	CreatedAt time.Time
}

func (s *Store) CreateUser(ctx context.Context, email, passwordHash string) (int64, error) {
	res, err := s.DB.ExecContext(ctx, `INSERT INTO users (email, password_hash) VALUES (?, ?)`, email, passwordHash)
	if err != nil {
		if isDup(err) {
			return 0, ErrConflict
		}
		return 0, err
	}
	return res.LastInsertId()
}

func (s *Store) GetUserByEmail(ctx context.Context, email string) (*User, string, error) {
	var u User
	var hash string
	err := s.DB.QueryRowContext(ctx, `SELECT id, email, password_hash, created_at FROM users WHERE email = ?`, email).
		Scan(&u.ID, &u.Email, &hash, &u.CreatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, "", ErrNotFound
	}
	return &u, hash, err
}

// ---------- conversations ----------

type Conversation struct {
	ID           string
	UserID       int64
	Title        string
	Summary      string
	SummaryUpto  int
	CreatedAt    time.Time
	UpdatedAt    time.Time
	MessageCount int
}

func (s *Store) CreateConversation(ctx context.Context, id string, userID int64, title string) error {
	if title == "" {
		title = "新会话"
	}
	_, err := s.DB.ExecContext(ctx,
		`INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)`, id, userID, title)
	return err
}

func (s *Store) ListConversations(ctx context.Context, userID int64) ([]Conversation, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT c.id, c.user_id, c.title, COALESCE(c.summary,''), c.summary_upto, c.created_at, c.updated_at,
		       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS cnt
		FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Conversation, 0, 32)
	for rows.Next() {
		var c Conversation
		if err := rows.Scan(&c.ID, &c.UserID, &c.Title, &c.Summary, &c.SummaryUpto, &c.CreatedAt, &c.UpdatedAt, &c.MessageCount); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (s *Store) GetConversation(ctx context.Context, id string) (*Conversation, error) {
	var c Conversation
	err := s.DB.QueryRowContext(ctx, `
		SELECT id, user_id, title, COALESCE(summary,''), summary_upto, created_at, updated_at
		FROM conversations WHERE id = ?`, id).
		Scan(&c.ID, &c.UserID, &c.Title, &c.Summary, &c.SummaryUpto, &c.CreatedAt, &c.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	return &c, nil
}

func (s *Store) RenameConversation(ctx context.Context, id string, userID int64, title string) error {
	_, err := s.DB.ExecContext(ctx, `UPDATE conversations SET title = ?, updated_at = NOW() WHERE id = ? AND user_id = ?`, title, id, userID)
	return err
}

func (s *Store) DeleteConversation(ctx context.Context, id string, userID int64) error {
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `DELETE FROM messages WHERE conversation_id = ?`, id); err != nil {
		return err
	}
	res, err := tx.ExecContext(ctx, `DELETE FROM conversations WHERE id = ? AND user_id = ?`, id, userID)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return ErrNotFound
	}
	return tx.Commit()
}

func (s *Store) TouchConversation(ctx context.Context, id string) error {
	_, err := s.DB.ExecContext(ctx, `UPDATE conversations SET updated_at = NOW() WHERE id = ?`, id)
	return err
}

func (s *Store) SetSummary(ctx context.Context, id, summary string, upto int) error {
	_, err := s.DB.ExecContext(ctx, `UPDATE conversations SET summary = ?, summary_upto = ? WHERE id = ?`, summary, upto, id)
	return err
}

// ---------- messages ----------

type Message struct {
	ID             int64
	ConversationID string
	Seq            int
	Role           string
	Content        string
	Reasoning      string
	ToolCalls      string
	ToolCallID     string
	ToolName       string
	Meta           string
	CreatedAt      time.Time
}

func (s *Store) AppendMessage(ctx context.Context, m Message) error {
	_, err := s.DB.ExecContext(ctx, `
		INSERT INTO messages (conversation_id, seq, role, content, reasoning, tool_calls, tool_call_id, tool_name, meta)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		m.ConversationID, m.Seq, m.Role, nullIfEmpty(m.Content), nullIfEmpty(m.Reasoning),
		nullIfEmpty(m.ToolCalls), nullIfEmpty(m.ToolCallID), nullIfEmpty(m.ToolName), nullIfEmpty(m.Meta))
	return err
}

func (s *Store) NextSeq(ctx context.Context, convID string) (int, error) {
	var seq sql.NullInt64
	if err := s.DB.QueryRowContext(ctx,
		`SELECT MAX(seq) FROM messages WHERE conversation_id = ?`, convID).Scan(&seq); err != nil {
		return 0, err
	}
	if seq.Valid {
		return int(seq.Int64) + 1, nil
	}
	return 1, nil
}

func (s *Store) ListMessages(ctx context.Context, convID string) ([]Message, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, conversation_id, seq, role, COALESCE(content,''), COALESCE(reasoning,''),
		       COALESCE(tool_calls,''), COALESCE(tool_call_id,''), COALESCE(tool_name,''), COALESCE(meta,''), created_at
		FROM messages WHERE conversation_id = ? ORDER BY seq ASC`, convID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.ConversationID, &m.Seq, &m.Role, &m.Content, &m.Reasoning,
			&m.ToolCalls, &m.ToolCallID, &m.ToolName, &m.Meta, &m.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// ---------- documents & chunks ----------

type Document struct {
	ID         string
	UserID     int64
	Name       string
	Size       int64
	ChunkCount int
	Status     string
	ErrMsg     string
	CreatedAt  time.Time
}

func (s *Store) CreateDocument(ctx context.Context, d Document) error {
	_, err := s.DB.ExecContext(ctx, `
		INSERT INTO documents (id, user_id, name, size, chunk_count, status, err_msg)
		VALUES (?, ?, ?, ?, ?, ?, ?)`, d.ID, d.UserID, d.Name, d.Size, d.ChunkCount, d.Status, nullIfEmpty(d.ErrMsg))
	return err
}

func (s *Store) UpdateDocumentStatus(ctx context.Context, id, status, errMsg string) error {
	_, err := s.DB.ExecContext(ctx, `UPDATE documents SET status = ?, err_msg = ? WHERE id = ?`, status, nullIfEmpty(errMsg), id)
	return err
}

func (s *Store) UpdateDocumentChunkCount(ctx context.Context, id string, n int) error {
	_, err := s.DB.ExecContext(ctx, `UPDATE documents SET chunk_count = ? WHERE id = ?`, n, id)
	return err
}

func (s *Store) ListDocuments(ctx context.Context, userID int64) ([]Document, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, user_id, name, size, chunk_count, status, COALESCE(err_msg,''), created_at
		FROM documents WHERE user_id = ? ORDER BY created_at DESC`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]Document, 0, 16)
	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.UserID, &d.Name, &d.Size, &d.ChunkCount, &d.Status, &d.ErrMsg, &d.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *Store) DeleteDocument(ctx context.Context, id string, userID int64) error {
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks WHERE document_id = ?`, id); err != nil {
		return err
	}
	res, err := tx.ExecContext(ctx, `DELETE FROM documents WHERE id = ? AND user_id = ?`, id, userID)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return ErrNotFound
	}
	return tx.Commit()
}

type Chunk struct {
	ID         int64
	DocumentID string
	UserID     int64
	Seq        int
	DocName    string
	Content    string
	Embedding  []byte
}

// InsertChunks 批量写入片段，返回与入参顺序一致的自增 ID，
// 该 ID 同时作为 Qdrant 的 point id，保证 MySQL 与向量库一一对应。
func (s *Store) InsertChunks(ctx context.Context, chunks []Chunk) ([]int64, error) {
	ids := make([]int64, 0, len(chunks))
	if len(chunks) == 0 {
		return ids, nil
	}
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	for _, c := range chunks {
		res, err := tx.ExecContext(ctx, `
			INSERT INTO chunks (document_id, user_id, seq, doc_name, content, embedding)
			VALUES (?, ?, ?, ?, ?, ?)`, c.DocumentID, c.UserID, c.Seq, c.DocName, c.Content, c.Embedding)
		if err != nil {
			return nil, err
		}
		id, err := res.LastInsertId()
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return ids, nil
}

func (s *Store) ListChunksByUser(ctx context.Context, userID int64) ([]Chunk, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, document_id, user_id, seq, doc_name, content, embedding
		FROM chunks WHERE user_id = ? ORDER BY document_id, seq`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.UserID, &c.Seq, &c.DocName, &c.Content, &c.Embedding); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// ListAllChunks 返回全部用户的片段，用于启动时把历史数据回填到向量库。
func (s *Store) ListAllChunks(ctx context.Context) ([]Chunk, error) {
	rows, err := s.DB.QueryContext(ctx, `
		SELECT id, document_id, user_id, seq, doc_name, content, embedding
		FROM chunks ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Chunk
	for rows.Next() {
		var c Chunk
		if err := rows.Scan(&c.ID, &c.DocumentID, &c.UserID, &c.Seq, &c.DocName, &c.Content, &c.Embedding); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (s *Store) CountUserChunks(ctx context.Context, userID int64) (int, error) {
	var n int
	err := s.DB.QueryRowContext(ctx, `SELECT COUNT(*) FROM chunks WHERE user_id = ?`, userID).Scan(&n)
	return n, err
}

// ---------- helpers ----------

func nullIfEmpty(v string) any {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	return v
}

func isDup(err error) bool {
	var me *mysql.MySQLError
	if errors.As(err, &me) {
		return me.Number == 1062
	}
	return false
}
