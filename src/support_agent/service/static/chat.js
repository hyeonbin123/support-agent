'use strict';

(() => {
  const LIMIT = 1000; // characters, counted as code points like the server does
  const COUNTER_FROM = 800;
  const POLL_MS = 5000;
  const STORAGE_KEY = 'support-agent.session-id';
  const THINKING = '답변 작성 중…';
  const ROLE_NAMES = { customer: '고객', agent: '상담원' };
  const NOTICES = {
    handoff: '상담원 연결을 요청했습니다. 이 대화는 종료되었습니다.',
    closed: '이 상담은 종료되었습니다. 이어서 도움이 필요하시면 새 상담을 시작해 주세요.',
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    scroll: $('scroll'),
    messages: $('messages'),
    progress: $('progress'),
    chips: $('chips'),
    notice: $('notice'),
    noticeText: $('notice-text'),
    error: $('error'),
    form: $('composer'),
    input: $('input'),
    counter: $('counter'),
    send: $('send'),
    hints: $('demo-hints'),
    hintsIntro: $('demo-hints-intro'),
  };

  let sessionId = null;
  let status = 'open';
  let busy = false; // a turn is running in this tab
  let polling = false;
  let epoch = 0; // grows with every new session; late answers of an older one are dropped
  let controller = null;
  let recoverable = null; // the shown error goes away by itself: 'load' on the next answer, 'reply' on new messages
  const displayed = []; // [{role, text}] in the order of the bubbles
  const pending = new Map(); // approval code -> {code, tool, refund_won}

  // ------------------------------------------------------------ rendering

  const won = (amount) => `${Number(amount).toLocaleString('ko-KR')}원`;
  const sessionUrl = (id) => `/api/sessions/${encodeURIComponent(id)}`;

  function remember(id) {
    try {
      if (id) sessionStorage.setItem(STORAGE_KEY, id);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // storage can be blocked; the chat still works for this page view
    }
  }

  function scrollToEnd() {
    els.scroll.scrollTop = els.scroll.scrollHeight;
  }

  function addBubble(role, text) {
    const bubble = document.createElement('div');
    bubble.className = `bubble ${role === 'customer' ? 'customer' : 'agent'}`;
    const speaker = document.createElement('span');
    speaker.className = 'sr-only';
    speaker.textContent = `${ROLE_NAMES[role] || role}: `;
    const body = document.createElement('span');
    body.textContent = text;
    bubble.append(speaker, body);
    els.messages.append(bubble);
    displayed.push({ role, text });
    scrollToEnd();
  }

  function truncate(length) {
    while (displayed.length > length) {
      displayed.pop();
      els.messages.lastElementChild.remove();
    }
  }

  // Append what is new. The server stores a turn only when it ends, so its list may lag behind the page
  // (then nothing changes); redraw only when the two lists disagree.
  function reconcile(messages) {
    const shared = Math.min(displayed.length, messages.length);
    const agree = displayed
      .slice(0, shared)
      .every((m, i) => m.role === messages[i].role && m.text === messages[i].text);
    if (!agree) truncate(0);
    for (const m of messages.slice(displayed.length)) addBubble(m.role, m.text);
  }

  function renderChips() {
    const items = [...pending.values()].map((a) => {
      const chip = document.createElement('li');
      chip.className = 'chip';
      chip.textContent = `담당자 승인 대기 중 (${a.code}, 환불 예정 ${won(a.refund_won)})`;
      return chip;
    });
    els.chips.replaceChildren(...items);
    if (items.length) scrollToEnd();
  }

  function setPending(list) {
    pending.clear();
    for (const a of list || []) pending.set(a.code, a);
    renderChips();
  }

  function setProgress(text) {
    els.progress.textContent = text;
    els.progress.hidden = !text;
    if (text) scrollToEnd();
  }

  function showError(text) {
    recoverable = null;
    els.error.textContent = text;
    els.error.hidden = !text;
  }

  function updateComposer() {
    const open = status === 'open' && sessionId !== null;
    const length = [...els.input.value.trim()].length;
    els.input.disabled = busy || !open;
    els.send.disabled = busy || !open || length === 0 || length > LIMIT;
    els.counter.hidden = length < COUNTER_FROM;
    els.counter.textContent = `${length.toLocaleString('ko-KR')} / ${LIMIT.toLocaleString('ko-KR')}`;
    els.counter.classList.toggle('over', length > LIMIT);
  }

  function renderStatus() {
    els.notice.hidden = status === 'open';
    els.noticeText.textContent = NOTICES[status] || NOTICES.closed;
    updateComposer();
  }

  function applyTranscript(transcript) {
    reconcile(transcript.messages || []);
    setPending(transcript.pending_approvals);
    status = transcript.status;
    renderStatus();
  }

  // ------------------------------------------------------------ sessions

  async function startSession({ focus = false } = {}) {
    epoch += 1;
    const mine = epoch;
    if (controller) controller.abort();
    busy = false;
    sessionId = null;
    status = 'open';
    remember(null);
    truncate(0);
    setPending([]);
    setProgress('');
    showError('');
    renderStatus();
    try {
      const response = await fetch('/api/sessions', { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const transcript = await response.json();
      if (mine !== epoch) return;
      sessionId = transcript.session_id;
      remember(sessionId);
      applyTranscript(transcript);
      if (focus) els.input.focus();
    } catch {
      if (mine === epoch) showError('상담을 시작하지 못했습니다. 잠시 후 "새 상담"을 눌러 다시 시도해 주세요.');
    }
  }

  // Bring the page in line with the server. Returns false when the server could not be asked.
  async function refresh({ fromPoll = false } = {}) {
    const mine = epoch;
    try {
      const response = await fetch(sessionUrl(sessionId));
      if (mine !== epoch || (fromPoll && busy)) return true;
      if (response.status === 404) {
        await startSession();
        showError('이전 상담을 찾을 수 없어 새 상담을 시작했습니다.');
        return true;
      }
      if (!response.ok) return false;
      const transcript = await response.json();
      // The server stores a turn when it ends, so a snapshot taken during a turn must not be applied.
      if (mine !== epoch || (fromPoll && busy)) return true;
      const before = displayed.length;
      applyTranscript(transcript);
      if (recoverable === 'load' || (recoverable === 'reply' && displayed.length > before)) showError('');
      return true;
    } catch {
      return false;
    }
  }

  async function poll() {
    if (busy || polling || !sessionId || document.hidden) return;
    if (status !== 'open' && pending.size === 0) return; // a decision can still arrive after a handoff
    polling = true;
    try {
      await refresh({ fromPoll: true });
    } finally {
      polling = false;
    }
  }

  // ------------------------------------------------------------ one turn

  function parseFrame(frame) {
    let event = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (!data.length) return null;
    try {
      return { event, data: JSON.parse(data.join('\n')) };
    } catch {
      return null;
    }
  }

  async function readStream(body, onFrame) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let cut;
      while ((cut = buffer.indexOf('\n\n')) !== -1) {
        const frame = parseFrame(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        if (frame) onFrame(frame.event, frame.data);
      }
      if (done) return;
    }
  }

  // Returns true for the last event of a turn.
  function onEvent(event, data) {
    if (event === 'status' || event === 'tool_result') setProgress(THINKING);
    else if (event === 'tool') setProgress(`${data.label || data.name} 중…`);
    else if (event === 'approval') {
      pending.set(data.code, data);
      renderChips();
    } else if (event === 'error') showError(data.message || '처리 중 문제가 생겼습니다.');
    else if (event === 'reply') {
      setProgress('');
      if (data.text) addBubble('agent', data.text);
    } else if (event === 'end') {
      status = data.status;
      renderStatus();
      return true;
    }
    return false;
  }

  function refusal(code) {
    if (code === 409) return '이 상담은 이미 종료되어 메시지를 보낼 수 없습니다.';
    if (code === 422) return `메시지는 1자 이상 ${LIMIT.toLocaleString('ko-KR')}자 이하로 입력해 주세요.`;
    return `메시지를 보내지 못했습니다. 잠시 후 다시 시도해 주세요. (오류 ${code})`;
  }

  async function send() {
    const text = els.input.value.trim();
    if (busy || status !== 'open' || !sessionId || !text || [...text].length > LIMIT) return;
    const mine = epoch;
    const index = displayed.length;
    busy = true;
    showError('');
    els.input.value = '';
    updateComposer();
    addBubble('customer', text);
    setProgress(THINKING);
    controller = new AbortController();
    let accepted = false; // the server took the message and a turn runs
    let ended = false;
    let replied = false;
    let sessionGone = false;
    try {
      const response = await fetch(`${sessionUrl(sessionId)}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (mine !== epoch) return;
      if (response.ok && response.body) {
        accepted = true;
        await readStream(response.body, (event, data) => {
          if (mine !== epoch) return;
          if (event === 'reply') replied = true;
          if (onEvent(event, data)) ended = true;
        });
        if (!ended) throw new Error('the stream stopped before its end event');
      } else if (response.status === 404) sessionGone = true;
      else showError(refusal(response.status));
    } catch {
      if (mine !== epoch) return; // a new session was started meanwhile
      if (accepted) {
        // The turn goes on at the server; polling shows its answer when it is stored.
        showError('연결이 끊겼습니다. 답변이 준비되면 이 화면에 자동으로 표시됩니다.');
        recoverable = 'reply';
      } else showError('메시지를 보내지 못했습니다. 네트워크를 확인하고 다시 시도해 주세요.');
    }
    if (mine !== epoch) return;
    setProgress('');
    if (sessionGone) {
      await startSession();
      showError('이전 상담을 찾을 수 없어 새 상담을 시작했습니다. 메시지를 다시 보내 주세요.');
    } else {
      // Not taken (refused, unreachable) or ended without an answer (the session was busy or not open).
      if (!accepted || (ended && !replied)) truncate(index);
      await refresh(); // busy stays true meanwhile, so a poll cannot interleave
      if (mine !== epoch) return;
      busy = false;
    }
    // Give the text back when the server did not record it (refused, busy, or never reached).
    const kept = displayed[index];
    const recorded = !sessionGone && kept && kept.role === 'customer' && kept.text === text;
    if (!recorded && !els.input.value) els.input.value = text;
    updateComposer();
    if (status === 'open') els.input.focus();
  }

  // ------------------------------------------------------------ demo hints

  async function loadHints() {
    try {
      const response = await fetch('/static/demo.json');
      if (!response.ok) return;
      const data = await response.json();
      for (const c of data.customers || []) {
        const item = document.createElement('li');
        const who = [c.name, c.phone].filter(Boolean).join(' · ');
        item.textContent = c.note ? `${who} — ${c.note}` : who;
        els.hints.append(item);
      }
      els.hintsIntro.hidden = els.hints.children.length === 0;
    } catch {
      // the file is optional
    }
  }

  // ------------------------------------------------------------ wiring

  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    send();
  });
  els.input.addEventListener('input', updateComposer);
  els.input.addEventListener('keydown', (event) => {
    // isComposing: Enter that only confirms a Hangul composition must not send
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    send();
  });
  $('new-session').addEventListener('click', () => startSession({ focus: true }));
  $('notice-new').addEventListener('click', () => startSession({ focus: true }));

  async function init() {
    loadHints();
    let saved = null;
    try {
      saved = sessionStorage.getItem(STORAGE_KEY);
    } catch {
      saved = null;
    }
    if (!saved) {
      await startSession();
    } else {
      sessionId = saved;
      if (!(await refresh())) {
        showError('상담 내용을 불러오지 못했습니다. 잠시 후 자동으로 다시 시도합니다.');
        recoverable = 'load';
      }
      updateComposer();
    }
    setInterval(poll, POLL_MS);
  }

  init();
})();
