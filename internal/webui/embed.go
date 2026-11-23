package webui

import "embed"

// Content 内嵌前端页面，保证二进制部署时无需额外文件。
//
//go:embed index.html app.js app.css
var Content embed.FS
