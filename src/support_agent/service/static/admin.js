'use strict';

(() => {
  const REFRESH_MS = 5000;
  const TOKEN_KEY = 'support-agent.admin-token';
  const BY_KEY = 'support-agent.admin-by';
  const APPROVAL_STATUS = {
    pending: '승인 대기',
    approved: '처리됨',
    rejected: '반려됨',
    failed: '승인했지만 처리 실패',
  };
  const SESSION_STATUS = { open: '진행 중', handoff: '상담원 연결', closed: '종료' };
  const ROLE_NAMES = { customer: '고객', agent: '상담원' };

  const $ = (id) => document.getElementById(id);
  const els = {
    login: $('login'),
    loginMessage: $('login-message'),
    tokenInput: $('token'),
    logout: $('logout'),
    app: $('app'),
    pendingCount: $('pending-count'),
    by: $('by'),
    showDecided: $('show-decided'),
    updated: $('approvals-updated'),
    result: $('approvals-result'),
    approvalsError: $('approvals-error'),
    approvalsEmpty: $('approvals-empty'),
    approvals: $('approvals'),
    sessionsError: $('sessions-error'),
    sessionsEmpty: $('sessions-empty'),
    sessions: $('sessions'),
    detail: $('detail'),
  };
  const tabs = { approvals: $('tab-approvals'), sessions: $('tab-sessions') };
  const panels = { approvals: $('panel-approvals'), sessions: $('panel-sessions') };

  let token = read(TOKEN_KEY);
  let approvals = [];
  let snapshot = ''; // what the approval list was drawn from; an unchanged answer keeps the DOM (and typing)
  let loadingApprovals = false;
  let deciding = null; // id of the approval whose decision is on its way
  let selectedSession = null;
  const notes = new Map(); // approval id -> note being typed

  // ------------------------------------------------------------ helpers

  function read(key) {
    try {
      return sessionStorage.getItem(key) || '';
    } catch {
      return '';
    }
  }

  function write(key, value) {
    try {
      if (value) sessionStorage.setItem(key, value);
      else sessionStorage.removeItem(key);
    } catch {
      // storage can be blocked; the value then lives for this page view only
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function say(node, text, tone) {
    node.textContent = text;
    node.hidden = !text;
    if (tone) node.dataset.tone = tone;
  }

  const won = (amount) => (amount == null ? '-' : `${Number(amount).toLocaleString('ko-KR')}원`);
  const timeFormat = new Intl.DateTimeFormat('ko-KR', {
    timeZone: 'Asia/Seoul',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  // Times are shown in KST. A value without an offset is taken as UTC (the database stores UTC).
  function time(iso) {
    if (!iso) return '-';
    const date = new Date(/(Z|[+-]\d\d:?\d\d)$/i.test(iso) ? iso : `${iso}Z`);
    return Number.isNaN(date.getTime()) ? iso : timeFormat.format(date);
  }

  class ApiError extends Error {
    constructor(status) {
      super(`HTTP ${status}`);
      this.status = status;
    }
  }
  const handled = (error) => error instanceof ApiError && (error.status === 401 || error.status === 503);

  async function api(path, body) {
    const options = { headers: { 'X-Admin-Token': token } };
    if (body !== undefined) {
      options.method = 'POST';
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    if (response.status === 401) {
      setToken('');
      showLogin('토큰이 올바르지 않습니다. 다시 입력해 주세요.');
    } else if (response.status === 503) showLogin('관리자 기능이 꺼져 있습니다.');
    if (!response.ok) throw new ApiError(response.status);
    return response.json();
  }

  // ------------------------------------------------------------ login

  function setToken(value) {
    token = value;
    write(TOKEN_KEY, value);
  }

  function showLogin(message) {
    els.app.hidden = true;
    els.logout.hidden = true;
    els.login.hidden = false;
    say(els.loginMessage, message || '');
    els.tokenInput.focus();
  }

  async function enter() {
    try {
      await api('/api/admin/approvals?status=pending');
    } catch (error) {
      if (!handled(error)) showLogin('서버에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
      return;
    }
    els.login.hidden = true;
    els.app.hidden = false;
    els.logout.hidden = false;
    snapshot = '';
    selectTab('approvals');
    loadApprovals();
  }

  // ------------------------------------------------------------ approval queue

  function approvalCard(a) {
    const card = el('article', `card ${a.status}`);
    const head = el('header', 'card-head');
    head.append(
      el('strong', '', a.code),
      el('span', `badge ${a.status}`, APPROVAL_STATUS[a.status] || a.status),
      el('span', 'muted', a.label || a.tool),
    );
    const facts = el('dl', 'facts');
    const rows = [
      ['주문', (a.args && a.args.order_id) || '-'],
      ['고객', a.customer_id || '-'],
      ['환불 예정', won(a.refund_won)],
      ['요청 시각', time(a.created_at)],
    ];
    if (a.status !== 'pending') {
      rows.push(['결정', `${time(a.decided_at)} · ${a.decided_by || '-'}`]);
      if (a.note) rows.push(['메모', a.note]);
    }
    for (const [name, value] of rows) facts.append(el('dt', '', name), el('dd', '', String(value)));
    card.append(head, facts);
    if (a.status === 'failed' && a.result) card.append(el('pre', 'payload', a.result));

    const actions = el('div', 'actions');
    if (a.status === 'pending') {
      const field = el('div', 'field grow');
      const label = el('label', '', '메모 (선택, 반려 사유는 고객에게 안내됩니다)');
      label.htmlFor = `note-${a.id}`;
      const note = el('input');
      note.type = 'text';
      note.id = label.htmlFor;
      note.maxLength = 500;
      note.autocomplete = 'off';
      note.dataset.noteFor = String(a.id);
      note.value = notes.get(a.id) || '';
      note.addEventListener('input', () => notes.set(a.id, note.value));
      field.append(label, note);
      const approve = el('button', 'btn btn-primary', '승인');
      const reject = el('button', 'btn btn-danger', '반려');
      for (const [button, value] of [[approve, true], [reject, false]]) {
        button.type = 'button';
        button.disabled = deciding !== null;
        button.addEventListener('click', () => decide(a, value));
      }
      actions.append(field, approve, reject);
    }
    const open = el('button', 'btn', '상담 보기');
    open.type = 'button';
    open.addEventListener('click', () => {
      selectTab('sessions');
      openSession(a.session_id);
    });
    actions.append(open);
    card.append(actions);
    return card;
  }

  function renderApprovals() {
    const active = document.activeElement;
    const focused = active && active.dataset ? active.dataset.noteFor : undefined;
    els.approvals.replaceChildren(...approvals.map(approvalCard));
    els.approvalsEmpty.hidden = approvals.length > 0;
    if (focused) {
      const again = els.approvals.querySelector(`[data-note-for="${Number(focused)}"]`);
      if (again) again.focus();
    }
  }

  async function loadApprovals() {
    if (!token || loadingApprovals) return;
    loadingApprovals = true;
    try {
      const all = els.showDecided.checked;
      const list = await api(`/api/admin/approvals${all ? '' : '?status=pending'}`);
      list.sort((a, b) => (b.status === 'pending') - (a.status === 'pending') || b.id - a.id);
      const count = list.filter((a) => a.status === 'pending').length;
      els.pendingCount.textContent = String(count);
      els.pendingCount.hidden = count === 0;
      const next = JSON.stringify([list, deciding]);
      if (next !== snapshot) {
        snapshot = next;
        approvals = list;
        renderApprovals();
      }
      els.updated.textContent = `${time(new Date().toISOString())} 갱신`;
      say(els.approvalsError, '');
    } catch (error) {
      if (!handled(error)) say(els.approvalsError, '승인 목록을 불러오지 못했습니다. 자동으로 다시 시도합니다.');
    } finally {
      loadingApprovals = false;
    }
  }

  async function decide(a, approve) {
    if (deciding !== null) return;
    deciding = a.id;
    renderApprovals(); // disables the buttons
    const by = els.by.value.trim() || 'admin';
    const note = (notes.get(a.id) || '').trim();
    try {
      const updated = await api(`/api/admin/approvals/${a.id}/decision`, { approve, by, note });
      notes.delete(a.id);
      const label = APPROVAL_STATUS[updated.status] || updated.status;
      say(els.result, `${updated.code}: ${label}`, updated.status === 'approved' ? 'ok' : 'warn');
    } catch (error) {
      if (handled(error)) return;
      const conflict = error instanceof ApiError && error.status === 409;
      say(
        els.result,
        conflict
          ? `${a.code}: 이미 처리되었거나 상담이 진행 중입니다. 잠시 후 다시 시도해 주세요.`
          : `${a.code}: 결정을 저장하지 못했습니다.`,
        'warn',
      );
    } finally {
      deciding = null;
      snapshot = '';
      if (token) {
        renderApprovals();
        await loadApprovals();
      }
    }
  }

  // ------------------------------------------------------------ sessions

  function bubble(message) {
    const node = el('div', `bubble ${message.role === 'customer' ? 'customer' : 'agent'}`);
    node.append(el('span', 'sr-only', `${ROLE_NAMES[message.role] || message.role}: `), el('span', '', message.text));
    return node;
  }

  function sessionItem(s) {
    const item = el('li');
    const button = el('button', 'session-item');
    button.type = 'button';
    button.dataset.session = s.session_id;
    if (s.session_id === selectedSession) button.setAttribute('aria-current', 'true');
    const top = el('span', 'session-top');
    top.append(
      el('code', '', `${s.session_id.slice(0, 10)}…`),
      el('span', `badge ${s.status}`, SESSION_STATUS[s.status] || s.status),
    );
    const info = `${s.turns}턴 · 고객 ${s.customer_id || '미확인'} · ${time(s.updated_at)}`;
    button.append(top, el('span', 'muted', info));
    button.addEventListener('click', () => openSession(s.session_id));
    item.append(button);
    return item;
  }

  async function loadSessions() {
    try {
      const list = await api('/api/admin/sessions');
      els.sessions.replaceChildren(...list.map(sessionItem));
      els.sessionsEmpty.hidden = list.length > 0;
      say(els.sessionsError, '');
    } catch (error) {
      if (!handled(error)) say(els.sessionsError, '상담 목록을 불러오지 못했습니다.');
    }
  }

  async function openSession(id) {
    selectedSession = id;
    for (const button of els.sessions.querySelectorAll('button')) {
      if (button.dataset.session === id) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    }
    let data;
    try {
      data = await api(`/api/admin/sessions/${encodeURIComponent(id)}`);
    } catch (error) {
      if (!handled(error)) say(els.sessionsError, '상담 내용을 불러오지 못했습니다.');
      return;
    }
    if (selectedSession !== id) return; // another one was picked meanwhile
    say(els.sessionsError, '');
    const transcript = data.transcript;
    $('detail-title').textContent = `상담 ${id.slice(0, 10)}…`;
    const state = SESSION_STATUS[transcript.status] || transcript.status;
    $('detail-meta').textContent = `상태: ${state} · 세션 ID: ${id}`;
    $('detail-messages').replaceChildren(...transcript.messages.map(bubble));
    $('detail-chips').replaceChildren(
      ...transcript.pending_approvals.map((p) => el('li', 'chip', `승인 대기 ${p.code} · 환불 예정 ${won(p.refund_won)}`)),
    );
    $('audit').replaceChildren(
      ...data.audit.map((event) => {
        const row = el('tr');
        const payload = el('td');
        payload.append(el('pre', 'payload', JSON.stringify(event.payload)));
        row.append(el('td', 'nowrap', time(event.at)), el('td', '', event.kind), payload);
        return row;
      }),
    );
    els.detail.hidden = false;
    els.detail.scrollIntoView({ block: 'nearest' });
  }

  // ------------------------------------------------------------ tabs and wiring

  function selectTab(name) {
    for (const key of Object.keys(tabs)) {
      const selected = key === name;
      tabs[key].setAttribute('aria-selected', String(selected));
      tabs[key].tabIndex = selected ? 0 : -1;
      panels[key].hidden = !selected;
    }
    if (name === 'sessions') loadSessions();
  }

  for (const [name, tab] of Object.entries(tabs)) {
    tab.addEventListener('click', () => selectTab(name));
    tab.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      const other = name === 'approvals' ? 'sessions' : 'approvals';
      selectTab(other);
      tabs[other].focus();
    });
  }

  $('login-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = els.tokenInput.value.trim();
    els.tokenInput.value = '';
    // A header value outside printable ASCII cannot be sent, so it cannot be the token either.
    if (!value || /[^\x20-\x7e]/.test(value)) {
      showLogin('토큰이 올바르지 않습니다. 다시 입력해 주세요.');
      return;
    }
    say(els.loginMessage, '');
    setToken(value);
    enter();
  });
  els.logout.addEventListener('click', () => {
    setToken('');
    showLogin('');
  });
  els.by.value = read(BY_KEY);
  els.by.addEventListener('input', () => write(BY_KEY, els.by.value.trim()));
  els.showDecided.addEventListener('change', loadApprovals);
  $('sessions-refresh').addEventListener('click', () => {
    loadSessions();
    if (selectedSession) openSession(selectedSession);
  });

  setInterval(() => {
    if (token && !els.app.hidden && !document.hidden) loadApprovals();
  }, REFRESH_MS);

  if (token) enter();
  else showLogin('');
})();
