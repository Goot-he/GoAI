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

type memEntry struct {
	data []byte
	exp  time.Time
}

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

func (c *Cache) Available() bool { return c != nil && c.rdb != nil }

func (c *Cache) Close() error {
	if c.rdb != nil {
		return c.rdb.Close()
	}
	return nil
}

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
