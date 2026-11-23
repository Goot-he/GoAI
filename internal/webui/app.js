const $ = (s) => document.querySelector(s);

const state = {
  token: localStorage.getItem('token') || '',
  user: null,
  convId: '',
  conversations: [],
  mode: 'login',
  busy: false,
  abort: null,
};

/* ---------- 基础请求 ---------- */

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (opts.body && !(opts.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    logout(true);
    throw new Error('登录已失效');
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || 'HTTP ' + res.status);
  return data;
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleString('zh-CN', { hour12: false }).replace(/\/./, '月').replace(/\/./, '日 ');
}

/* ---------- 登录 ---------- */

function showAuth() {
  $('#auth').classList.remove('hidden');
  $('#app').classList.add('hidden');
}

function showApp() {
  $('#auth').classList.add('hidden');
  $('#app').classList.remove('hidden');
  $('#user-email').textContent = state.user ? state.user.email : '';
  refreshConversations();
  loadStats();
}

function logout(silent) {
  state.token = '';
  state.user = null;
  localStorage.removeItem('token');
  showAuth();
  if (!silent) $('#auth-error').textContent = '';
}

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
    t.classList.add('active');
    state.mode = t.dataset.mode;
    $('#auth-submit').textContent = state.mode === 'login' ? '登录' : '注册';
  });
});

$('#auth-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = $('#email').value.trim();
  const password = $('#password').value;
  $('#auth-error').textContent = '';
  $('#auth-submit').disabled = true;
  try {
    const path = state.mode === 'login' ? '/api/auth/login' : '/api/auth/register';
    const data = await api(path, { method: 'POST', body: JSON.stringify({ email, password }) });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem('token', data.token);
    showApp();
  } catch (err) {
    $('#auth-error').textContent = err.message;
  } finally {
    $('#auth-submit').disabled = false;
  }
});

$('#logout').addEventListener('click', () => logout(false));

/* ---------- 会话列表 ---------- */

async function refreshConversations() {
  try {
    const data = await api('/api/conversations');
    state.conversations = data.conversations || [];
    renderConversations();
  } catch (err) {
    console.warn(err);
  }
}

function renderConversations() {
  const list = $('#conv-list');
  list.innerHTML = '';
  if (state.conversations.length === 0) {
    const li = el('li', 'muted small', '还没有历史会话');
    li.style.padding = '8px 4px';
    list.appendChild(li);
    return;
  }
  state.conversations.forEach((c) => {
    const li = el('li', 'conv-item' + (c.id === state.convId ? ' active' : ''));
    const row = el('div', 'row');
    const title = el('strong', '', c.title || '新会话');
    const del = el('button', 'del', '×');
    del.title = '删除会话';
    del.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      if (!confirm('删除该会话及其全部消息？')) return;
      try {
        await api('/api/conversations/' + c.id, { method: 'DELETE' });
        if (state.convId === c.id) newChat();
        refreshConversations();
        loadStats();
      } catch (err) {
        alert(err.message);
      }
    });
    row.append(title, del);
    const meta = el('span', '', `${c.message_count} 条消息 · ${fmtTime(c.updated_at)}`);
    li.append(row, meta);
    li.addEventListener('click', () => openConversation(c.id));
    list.appendChild(li);
  });
}

async function openConversation(id) {
  if (state.busy) return;
  state.convId = id;
  renderConversations();
  const box = $('#messages');
  box.innerHTML = '';
  try {
    const data = await api('/api/conversations/' + id + '/messages');
    $('#chat-title').textContent = data.conversation.title || '会话';
    $('#chat-sub').textContent = data.conversation.has_summary
      ? '已恢复历史会话（早期内容已摘要压缩，仍保留原文回看）'
      : '已恢复历史会话，可继续追问';
    (data.messages || []).forEach((m) => renderStoredMessage(m));
    scrollBottom(true);
  } catch (err) {
    box.innerHTML = '';
    box.appendChild(el('div', 'empty', err.message));
  }
}

function renderStoredMessage(m) {
  if (m.role === 'user') {
    addUserMessage(m.content, m.created_at);
    return;
  }
  if (m.role !== 'assistant') return;
  // 中间轮（带 tool_calls）只作为过程，最终回答单独渲染
  if (m.tool_calls && m.tool_calls.length) return;
  if (!m.content) return;

  const ui = createAssistantMessage({ live: false });
  const steps = (m.meta && m.meta.steps) || [];
  const hasThinking = steps.length > 0 || (m.reasoning || '');
  if (hasThinking && $('#think-toggle').checked) {
    ui.showThinking();
    steps.forEach((s) => {
      if (s.type === 'tool_call') ui.addToolCall(s.tool, s.args);
      else if (s.type === 'tool_result') ui.addToolResult(s.tool, s.text);
      else if (s.type === 'status') ui.addStatus(s.text);
    });
    // 旧数据没有“结果”状态，补一条，保证思考面板不会停在“正在理解问题…”
    const hasResult = steps.some((s) => s.type === 'status' && (s.text || '').indexOf('结果如下') === 0);
    if (!hasResult) ui.addStatus('结果如下');
    if (m.reasoning) ui.addReasoning(m.reasoning);
  }
  ui.setContent(m.content || '');
  const usage = (m.meta && m.meta.usage) || null;
  const facts = [];
  if (usage && usage.total_tokens) facts.push('tokens ' + usage.total_tokens);
  if (steps.length) facts.push('工具步骤 ' + steps.filter((s) => s.type === 'tool_call').length);
  if (m.created_at) facts.push(fmtTime(m.created_at));
  ui.setMeta(facts.join(' · '));
  ui.finish();
}

function newChat() {
  if (state.busy) return;
  state.convId = '';
  $('#messages').innerHTML = '';
  $('#chat-title').textContent = '新会话';
  $('#chat-sub').textContent = '上传文档后提问，回答会标注来源';
  const hint = el('div', 'empty');
  hint.innerHTML = '<h2>开始一次基于文档的对话</h2><p>先在「知识库管理」上传 .md / .txt / .docx，再提问。</p>';
  $('#messages').appendChild(hint);
  renderConversations();
}

/* ---------- 消息渲染 ---------- */

function addUserMessage(text, time) {
  const empty = $('#empty-hint');
  if (empty) empty.remove();
  const root = el('div', 'msg user');
  const av = el('div', 'avatar', '我');
  const bubble = el('div', 'bubble');
  const content = el('div', 'content');
  content.innerHTML = mdToHtml(text);
  bubble.appendChild(content);
  if (time) {
    const meta = el('div', 'meta');
    meta.appendChild(el('span', '', fmtTime(time)));
    bubble.appendChild(meta);
  }
  root.append(av, bubble);
  $('#messages').appendChild(root);
  scrollBottom();
}

function createAssistantMessage({ live = true } = {}) {
  const root = el('div', 'msg assistant');
  const av = el('div', 'avatar', 'AI');
  const bubble = el('div', 'bubble');

  const think = el('div', 'thinking hidden');
  const head = el('div', 'thinking-head');
  const dot = el('span', 'dot');
  const label = el('span', '', '思考过程');
  const arrow = el('span', 'arrow', '展开');
  head.append(dot, label, arrow);

  const body = el('div', 'thinking-body');
  body.style.display = 'none';

  const reasoningBox = el('div', 'reasoning-text');
  reasoningBox.style.display = 'none';

  think.append(head, body);
  body.appendChild(reasoningBox);

  const content = el('div', 'content');
  const typing = el('div', 'typing', '正在思考…');
  typing.style.display = $('#think-toggle').checked ? 'none' : 'block';
  const meta = el('div', 'meta');
  meta.style.display = 'none';

  bubble.append(think, typing, content, meta);
  root.append(av, bubble);
  $('#messages').appendChild(root);

  const empty = $('#empty-hint');
  if (empty) empty.remove();

  let reasoningText = '';
  let expanded = false;
  const showThink = $('#think-toggle').checked;
  const setExpanded = (v) => {
    expanded = v;
    body.style.display = v ? 'grid' : 'none';
    arrow.textContent = v ? '收起' : '展开';
  };
  head.addEventListener('click', () => setExpanded(!expanded));

  const addItem = (tagClass, tagText, text) => {
    if (!showThink) return;
    think.classList.remove('hidden');
    if (!expanded) setExpanded(true);
    const item = el('div', 't-item');
    const tag = el('span', 'tag ' + tagClass, tagText);
    const tx = el('span', '', text);
    item.append(tag, tx);
    body.appendChild(item);
  };

  const ui = {
    root,
    showThinking() {
      if (!showThink) return;
      think.classList.remove('hidden');
      if (!expanded) setExpanded(true);
    },
    addStatus(text) {
      const isResult = (text || '').indexOf('结果如下') === 0;
      addItem(isResult ? 'done' : '', isResult ? '结果' : '状态', text);
      scrollBottom();
    },
    addToolCall(tool, args) {
      addItem('tool', '工具', `调用 ${tool}${args ? '：' + args : ''}`);
      scrollBottom();
    },
    addToolResult(tool, text) {
      addItem('tool', '结果', `【${tool}】${text || '已完成'}`);
      scrollBottom();
    },
    addReasoning(delta) {
      typing.remove();
      if (!showThink) return;
      think.classList.remove('hidden');
      reasoningText += delta;
      reasoningBox.style.display = 'block';
      reasoningBox.textContent = reasoningText;
      if (!expanded) setExpanded(true);
      scrollBottom();
    },
    appendDelta(delta) {
      typing.remove();
      content.textContent += delta;
      content.innerHTML = mdToHtml(content.textContent);
      scrollBottom();
    },
    setContent(text) {
      typing.remove();
      content.innerHTML = mdToHtml(text || '');
    },
    setMeta(text) {
      if (!text) return;
      meta.style.display = 'flex';
      meta.innerHTML = '';
      text.split('·').forEach((p) => {
        const s = p.trim();
        if (s) meta.appendChild(el('span', '', s));
      });
    },
    finish() {
      typing.remove();
      head.querySelector('.dot').style.animation = 'none';
      head.querySelector('.dot').style.opacity = '.5';
      label.textContent = '思考过程（已完成）';
      if (!live) {
        setExpanded(false); // 恢复的历史会话默认折叠，保持界面整洁
        return;
      }
      if ($('#think-toggle').checked && body.childElementCount > 1) {
        // 保持展开，便于用户检查依据
      } else {
        setExpanded(false);
      }
    },
    error(msg) {
      typing.remove();
      content.innerHTML = '<span style="color:var(--danger)">' + escapeHtml(msg) + '</span>';
    },
  };

  if (!$('#think-toggle').checked) think.classList.add('hidden');
  return ui;
}

function mdToHtml(text) {
  if (!text) return '';
  const escaped = escapeHtml(text);
  const parts = escaped.split('```');
  let out = '';
  parts.forEach((p, i) => {
    if (i % 2 === 1) {
      const nl = p.indexOf('\n');
      let code = p;
      if (nl >= 0) code = p.slice(nl + 1);
      out += '<pre><code>' + code.replace(/\n$/, '') + '</code></pre>';
    } else {
      let seg = p
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      out += seg;
    }
  });
  return out;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function scrollBottom(force) {
  const box = $('#messages');
  const near = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  if (force || near) box.scrollTop = box.scrollHeight;
}

/* ---------- 发送 ---------- */

$('#input').addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 180) + 'px';
});

$('#input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

$('#send').addEventListener('click', () => {
  if (state.busy && state.abort) {
    state.abort.abort();
    return;
  }
  send();
});

document.querySelectorAll('.sample').forEach((b) => {
  b.addEventListener('click', () => {
    $('#input').value = b.textContent;
    send();
  });
});

async function send() {
  if (state.busy) return;
  const text = $('#input').value.trim();
  if (!text) return;
  $('#input').value = '';
  $('#input').style.height = 'auto';

  addUserMessage(text);
  const assistant = createAssistantMessage();
  const started = Date.now();
  state.busy = true;
  $('#send').textContent = '停止';
  const ctrl = new AbortController();
  state.abort = ctrl;

  try {
    if ($('#stream-toggle').checked) {
      await streamChat(text, assistant, ctrl.signal, started);
    } else {
      await plainChat(text, assistant, started);
    }
  } catch (err) {
    if (err.name !== 'AbortError') assistant.error(err.message);
  } finally {
    state.busy = false;
    state.abort = null;
    $('#send').textContent = '发送';
    assistant.finish();
    refreshConversations();
    loadStats();
  }
}

async function streamChat(text, ui, signal, started) {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + state.token },
    body: JSON.stringify({ conversation_id: state.convId, message: text }),
    signal,
  });
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    throw new Error(d.error || 'HTTP ' + res.status);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let usage = null;
  let steps = 0;

  const handle = (raw) => {
    let event = 'message';
    const dataLines = [];
    raw.split('\n').forEach((line) => {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    });
    let payload = {};
    try {
      payload = JSON.parse(dataLines.join('\n') || '{}');
    } catch (_) {}
    switch (event) {
      case 'meta':
        if (payload.conversation_id) state.convId = payload.conversation_id;
        if (payload.text) $('#chat-title').textContent = payload.text;
        break;
      case 'status':
        ui.addStatus(payload.text || '');
        break;
      case 'reasoning':
        ui.addReasoning(payload.text || '');
        break;
      case 'delta':
        ui.appendDelta(payload.text || '');
        break;
      case 'tool_call':
        steps++;
        ui.addToolCall(payload.tool, payload.args);
        break;
      case 'tool_result':
        ui.addToolResult(payload.tool, payload.text);
        break;
      case 'usage':
        usage = payload;
        break;
      case 'done':
        $('#chat-sub').textContent = '回答完成，可继续追问或打开思考过程查看依据';
        break;
      case 'error':
        ui.error(payload.text || '未知错误');
        break;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (chunk.trim()) handle(chunk);
    }
  }

  const facts = [];
  facts.push('用时 ' + ((Date.now() - started) / 1000).toFixed(1) + 's');
  if (steps) facts.push('工具步骤 ' + steps);
  if (usage && usage.total_tokens) facts.push('tokens ' + usage.total_tokens);
  ui.setMeta(facts.join(' · '));
}

async function plainChat(text, ui, started) {
  ui.addStatus('非流式模式：等待模型完整返回…');
  const data = await api('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ conversation_id: state.convId, message: text }),
  });
  if (data.conversation_id) state.convId = data.conversation_id;
  if (data.title) $('#chat-title').textContent = data.title;

  const steps = data.steps || [];
  if ($('#think-toggle').checked) {
    ui.showThinking();
    steps.forEach((s) => {
      if (s.type === 'tool_call') ui.addToolCall(s.tool, s.args);
      else if (s.type === 'tool_result') ui.addToolResult(s.tool, s.text);
      else if (s.type === 'status') ui.addStatus(s.text);
    });
  }
  if (data.reasoning) ui.addReasoning(data.reasoning);
  ui.setContent(data.content || '（模型未返回内容）');

  const facts = ['用时 ' + ((Date.now() - started) / 1000).toFixed(1) + 's'];
  if (steps.filter((s) => s.type === 'tool_call').length) {
    facts.push('工具步骤 ' + steps.filter((s) => s.type === 'tool_call').length);
  }
  if (data.usage && data.usage.total_tokens) facts.push('tokens ' + data.usage.total_tokens);
  ui.setMeta(facts.join(' · '));
}

/* ---------- 知识库 ---------- */

$('#open-docs').addEventListener('click', () => {
  $('#docs-modal').classList.remove('hidden');
  loadDocs();
});
$('#close-docs').addEventListener('click', () => $('#docs-modal').classList.add('hidden'));
$('#docs-modal').addEventListener('click', (e) => {
  if (e.target === $('#docs-modal')) $('#docs-modal').classList.add('hidden');
});

$('#file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const status = $('#upload-status');
  status.textContent = '上传并向量化中：' + file.name + ' …';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await api('/api/documents', { method: 'POST', body: fd });
    status.textContent = `已入库：${data.name}，切分为 ${data.chunk_count} 个片段`;
    loadDocs();
    loadStats();
  } catch (err) {
    status.textContent = '失败：' + err.message;
  } finally {
    e.target.value = '';
  }
});

async function loadDocs() {
  const list = $('#doc-list');
  try {
    const data = await api('/api/documents');
    list.innerHTML = '';
    if (!data.documents.length) {
      list.appendChild(el('li', 'muted small', '还没有文档，先上传一份试试'));
      return;
    }
    data.documents.forEach((d) => {
      const li = el('li', 'doc-item');
      const info = el('div', 'info');
      info.appendChild(el('strong', '', d.name));
      info.appendChild(el('span', '', `${(d.size / 1024).toFixed(1)} KB · ${d.chunk_count} 片段 · ${fmtTime(d.created_at)}`));
      const badge = el('span', 'badge ' + d.status, d.status === 'ready' ? '已就绪' : d.status === 'failed' ? '失败' : '处理中');
      if (d.error) badge.title = d.error;
      const rm = el('button', 'rm', '删除');
      rm.addEventListener('click', async () => {
        if (!confirm('删除文档《' + d.name + '》及其索引？')) return;
        try {
          await api('/api/documents/' + d.id, { method: 'DELETE' });
          loadDocs();
          loadStats();
        } catch (err) {
          alert(err.message);
        }
      });
      li.append(info, badge, rm);
      list.appendChild(li);
    });
  } catch (err) {
    list.innerHTML = '';
    list.appendChild(el('li', 'muted small', err.message));
  }
}

/* ---------- 统计与启动 ---------- */

async function loadStats() {
  try {
    const s = await api('/api/stats');
    const box = $('#stats');
    box.innerHTML = '';
    const lines = [
      '模型：' + s.model,
      '向量：' + s.embedder + (s.redis ? ' · Redis' : ' · 内存缓存'),
      '文档 ' + s.documents + ' · 片段 ' + s.chunks + ' · 会话 ' + s.conversations,
    ];
    lines.forEach((t) => box.appendChild(el('div', '', t)));
  } catch (_) {}
}

$('#stream-toggle').addEventListener('change', () => {
  $('#chat-sub').textContent = $('#stream-toggle').checked
    ? '流式输出：逐字展示，并实时显示思考与工具调用'
    : '非流式输出：等待完整回答后一次性展示';
});

async function boot() {
  if (!state.token) {
    showAuth();
    return;
  }
  try {
    const data = await api('/api/auth/me');
    state.user = data.user;
    showApp();
  } catch (_) {
    showAuth();
  }
}

boot();
