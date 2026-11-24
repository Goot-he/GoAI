package cache

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache 优先使用 Redis；不可用时退化为进程内 TTL 缓存，保证功能不中断。
// 对应课程 M09/M11 的思路：缓存是优化层，不能成为可用性的单点。
type Cache struct {
	rdb *redis.Client

	mu  sync.Mutex
	mem map[string]memEntry
}

// memEntry 是进程内缓存条目：序列化后的数据与过期时间。
type memEntry struct {
	data []byte
	exp  time.Time
}

// New 初始化缓存：Redis 可用则用 Redis，Ping 失败或地址为空时退回进程内 TTL 缓存。
func New(ctx context.Context, addr, password string, db int) *Cache {
	c := &Cache{mem: make(map[string]memEntry)}
	if addr == "" {
		return c
	}
	rdb := redis.NewClient(&redis.Options{Addr: addr, Password: password, DB: db})
	pingCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if err := rdb.Ping(pingCtx).Err(); err != nil {
		_ = rdb.Close()
		return c
	}
	c.rdb = rdb
	return c
}

// Available 返回当前是否真正连上了 Redis（false 表示在用进程内缓存）。
func (c *Cache) Available() bool { return c != nil && c.rdb != nil }

// Close 关闭 Redis 连接，未启用 Redis 时为空操作。
func (c *Cache) Close() error {
	if c.rdb != nil {
		return c.rdb.Close()
	}
	return nil
}

// GetJSON 读取缓存并反序列化到 out，未命中或解析失败返回 false。
func (c *Cache) GetJSON(ctx context.Context, key string, out any) bool {
	if c == nil {
		return false
	}
	if c.rdb != nil {
		data, err := c.rdb.Get(ctx, key).Bytes()
		if err != nil || len(data) == 0 {
			return false
		}
		return json.Unmarshal(data, out) == nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.mem[key]
	if !ok || time.Now().After(e.exp) {
		if ok {
			delete(c.mem, key)
		}
		return false
	}
	return json.Unmarshal(e.data, out) == nil
}

// SetJSON 序列化后写入缓存并设置 TTL；内存模式会在条目过多时清理过期数据。
func (c *Cache) SetJSON(ctx context.Context, key string, v any, ttl time.Duration) {
	if c == nil {
		return
	}
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	if c.rdb != nil {
		_ = c.rdb.Set(ctx, key, data, ttl).Err()
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.mem) > 5000 { // 简单的容量上限，避免内存无界增长
		for k, e := range c.mem {
			if time.Now().After(e.exp) {
				delete(c.mem, k)
			}
		}
		if len(c.mem) > 5000 {
			c.mem = make(map[string]memEntry)
		}
	}
	c.mem[key] = memEntry{data: data, exp: time.Now().Add(ttl)}
}

// Delete 删除若干缓存键，用于文档变更后失效相关缓存。
func (c *Cache) Delete(ctx context.Context, keys ...string) {
	if c == nil || len(keys) == 0 {
		return
	}
	if c.rdb != nil {
		_ = c.rdb.Del(ctx, keys...).Err()
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, k := range keys {
		delete(c.mem, k)
	}
}
