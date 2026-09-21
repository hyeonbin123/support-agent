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
    // voice: hidden unless the server has it
    voiceBar: $('voice-bar'),
    speak: $('speak-replies'),
    speakStop: $('speak-stop'),
    recStatus: $('rec-status'),
    recCancel: $('rec-cancel'),
    mic: $('mic'),
  };

  let sessionId = null;
  let status = 'open';
  let busy = false; // a turn is running in this tab
  let rec = null; // the recording in progress (see voice); typing, sending and polling wait for it
  let polling = false;
  let epoch = 0; // grows with every new session; late answers of an older one are dropped
  let controller = null;
  let recoverable = null; // the shown error goes away by itself: 'load' on the next answer, 'reply' on new messages
  const displayed = []; // [{role, text}] in the order of the bubbles
  const pending = new Map(); // approval code -> {code, tool, refund_won}

  // ------------------------------------------------------------ rendering

  const won = (amount) => `${Number(amount).toLocaleString('ko-KR')}원`;
  const sessionUrl = (id) => `/api/sessions/${encodeURIComponent(id)}`;

  function remember(value, key = STORAGE_KEY) {
    try {
      if (value) sessionStorage.setItem(key, value);
      else sessionStorage.removeItem(key);
    } catch {
      // storage can be blocked; the chat still works for this page view
    }
  }

  function recall(key = STORAGE_KEY) {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
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
    els.input.disabled = busy || !open || rec !== null;
    els.send.disabled = els.input.disabled || length === 0 || length > LIMIT;
    els.counter.hidden = length < COUNTER_FROM;
    els.counter.textContent = `${length.toLocaleString('ko-KR')} / ${LIMIT.toLocaleString('ko-KR')}`;
    els.counter.classList.toggle('over', length > LIMIT);
    renderMic(open);
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
    stopRecording(false);
    stopSpeaking();
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
      if (mine !== epoch || (fromPoll && (busy || rec))) return true;
      if (response.status === 404) {
        await startSession();
        showError('이전 상담을 찾을 수 없어 새 상담을 시작했습니다.');
        return true;
      }
      if (!response.ok) return false;
      const transcript = await response.json();
      // The server stores a turn when it ends, so a snapshot taken during a turn must not be applied
      // (nor during a recording: a message that is read aloud would be recorded).
      if (mine !== epoch || (fromPoll && (busy || rec))) return true;
      const before = displayed.slice();
      const waiting = [...pending.keys()];
      applyTranscript(transcript);
      if (recoverable === 'load' || (recoverable === 'reply' && displayed.length > before.length)) showError('');
      // An approval was decided: its message is new, unlike the rest of a stored conversation, so it is
      // read aloud like a reply.
      if (waiting.some((code) => !pending.has(code))) unseenReplies(before).forEach((text) => speak(text));
      return true;
    } catch {
      return false;
    }
  }

  async function poll() {
    if (busy || rec || polling || !sessionId || document.hidden) return;
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
      if (data.text) {
        addBubble('agent', data.text);
        speak(data.text);
      }
    } else if (event === 'end') {
      status = data.status;
      renderStatus();
      return true;
    }
    return false;
  }

  function refusal(code, spoken) {
    if (spoken && VOICE_REFUSALS[code]) return VOICE_REFUSALS[code];
    if (code === 409) return '이 상담은 이미 종료되어 메시지를 보낼 수 없습니다.';
    if (code === 422) return `메시지는 1자 이상 ${LIMIT.toLocaleString('ko-KR')}자 이하로 입력해 주세요.`;
    return `메시지를 보내지 못했습니다. 잠시 후 다시 시도해 주세요. (오류 ${code})`;
  }

  function send() {
    const text = els.input.value.trim();
    if (text && [...text].length <= LIMIT) runTurn(text);
  }

  // One turn. A typed message is shown at once; what a recording (`audio`) says is known when `heard` arrives.
  async function runTurn(typed, audio = null) {
    if (busy || rec || status !== 'open' || !sessionId) return;
    let text = typed;
    const mine = epoch;
    const index = displayed.length;
    busy = true;
    showError('');
    if (!audio) els.input.value = '';
    updateComposer();
    if (!audio) addBubble('customer', text);
    setProgress(audio ? HEARING : THINKING);
    controller = new AbortController();
    let accepted = false; // the server took the message and a turn runs
    let ended = false;
    let replied = false;
    let sessionGone = false;
    try {
      const response = await fetch(`${sessionUrl(sessionId)}/${audio ? 'voice' : 'messages'}`, {
        method: 'POST',
        headers: {
          'Content-Type': audio ? audio.type || 'application/octet-stream' : 'application/json',
          Accept: 'text/event-stream',
        },
        body: audio || JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (mine !== epoch) return;
      if (response.ok && response.body) {
        accepted = true;
        await readStream(response.body, (event, data) => {
          if (mine !== epoch) return;
          if (event === 'heard') {
            text = (data.text || '').trim() || null; // nothing understood: an `error` follows, no bubble
            if (!text) return;
            addBubble('customer', text);
            setProgress(THINKING);
            if (!speakChosen) setSpeaking(true); // whoever talks most likely wants to listen
            return;
          }
          if (event === 'status' && audio && !text) return; // still recognising the speech
          if (event === 'reply') replied = true;
          if (onEvent(event, data)) ended = true;
        });
        if (!ended) throw new Error('the stream stopped before its end event');
      } else if (response.status === 404) sessionGone = true;
      else {
        showError(refusal(response.status, Boolean(audio)));
        if (audio && response.status === 503) setVoice(false);
      }
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
    if (text && !recorded && !els.input.value) els.input.value = text;
    updateComposer();
    // After a recording the microphone button keeps the focus: on a phone the text box would open the keyboard.
    if (status === 'open') (audio && voiceOn ? els.mic : els.input).focus();
  }

  // ------------------------------------------------------------ voice (optional)
  // On only when /healthz reports `voice`. A recording goes through runTurn() like a typed message; the
  // replies of a turn are read aloud one after another when the customer wants to listen.

  const MIC_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  const MAX_RECORD_MS = 30000;
  const MIN_RECORD_MS = 300; // shorter than this is a slip of the finger
  const MAX_AUDIO_BYTES = 5000000; // the server's default limit; above its own it answers 413
  const SPEECH_LIMIT = 600; // characters in one /speech request
  const SPEAK_KEY = 'support-agent.speak-replies';
  const HEARING = '음성 인식 중…';
  const MIC_LABEL = '음성으로 말하기';
  const MIC_START = '말하기';
  const MIC_STOP = '녹음 중… 눌러서 전송';
  const NO_MIC = '마이크를 사용할 수 없습니다. 브라우저의 마이크 권한을 확인해 주세요.';
  const NO_AUTOPLAY = '브라우저가 소리 재생을 막았습니다. 화면을 한 번 누르면 다음 답변부터 읽어 드립니다.';
  const VOICE_REFUSALS = {
    413: '녹음이 너무 깁니다. 짧게 나누어 다시 말씀해 주세요.',
    422: '녹음된 소리가 없습니다. 다시 말씀해 주세요.',
    503: '지금은 음성 기능을 사용할 수 없습니다. 메시지를 글로 입력해 주세요.',
  };

  let voiceOn = false;
  let speakChosen = false; // the listening toggle was set, by the customer or by the first voice message
  const speech = { queue: [], run: 0, busy: false, audio: null, abort: null, finish: null };

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  // Hide a button without losing the keyboard's place.
  function hideButton(button, next) {
    if (document.activeElement === button) next.focus();
    button.hidden = true;
  }

  // --- recording: `rec` is {stream, recorder, chunks, startedAt, timer, ending}

  function renderMic(open) {
    const live = Boolean(rec && rec.startedAt); // before that the browser is asking for the microphone
    els.mic.disabled = busy || !open || Boolean(rec && rec.ending);
    els.mic.classList.toggle('btn-danger', live);
    if (els.mic.textContent !== (live ? MIC_STOP : MIC_START)) {
      els.mic.textContent = live ? MIC_STOP : MIC_START;
      els.mic.setAttribute('aria-label', live ? MIC_STOP : MIC_LABEL);
    }
    els.voiceBar.classList.toggle('recording', rec !== null);
    els.recStatus.hidden = !rec;
    els.recStatus.classList.toggle('live', live);
    if (!live) setText(els.recStatus, rec ? '마이크 연결 중…' : '');
    if (rec) els.recCancel.hidden = false;
    else hideButton(els.recCancel, els.mic);
  }

  function tick() {
    if (!rec || !rec.startedAt || rec.ending) return;
    const elapsed = Date.now() - rec.startedAt;
    if (elapsed >= MAX_RECORD_MS) stopRecording(true);
    else setText(els.recStatus, `녹음 중 ${Math.floor(elapsed / 1000)}초 / ${MAX_RECORD_MS / 1000}초`);
  }

  function release(take) {
    clearInterval(take.timer);
    if (take.recorder && take.recorder.state !== 'inactive') take.recorder.stop();
    if (take.stream) for (const track of take.stream.getTracks()) track.stop(); // the browser's mic sign goes off
  }

  async function startRecording() {
    if (rec || busy || status !== 'open' || !sessionId) return;
    stopSpeaking(); // it would be recorded
    showError('');
    const media = navigator.mediaDevices;
    if (!media || !media.getUserMedia || typeof MediaRecorder === 'undefined') {
      showError(NO_MIC);
      return;
    }
    const mine = { stream: null, recorder: null, chunks: [], startedAt: 0, timer: 0, ending: false };
    rec = mine;
    updateComposer();
    try {
      mine.stream = await media.getUserMedia({ audio: true });
      if (rec !== mine) {
        release(mine); // cancelled while the browser asked for permission
        return;
      }
      const type = MIC_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
      mine.recorder = type ? new MediaRecorder(mine.stream, { mimeType: type }) : new MediaRecorder(mine.stream);
      mine.recorder.ondataavailable = (event) => {
        if (event.data && event.data.size) mine.chunks.push(event.data);
      };
      mine.recorder.onstop = () => finishRecording(mine);
      mine.recorder.start();
    } catch {
      release(mine); // no permission, no microphone, or a recorder that does not start
      if (rec !== mine) return;
      rec = null;
      showError(NO_MIC);
      updateComposer();
      return;
    }
    mine.startedAt = Date.now();
    mine.timer = setInterval(tick, 250);
    updateComposer();
    tick();
  }

  // Turn the microphone off. `keep`: send what was recorded; otherwise it is thrown away.
  function stopRecording(keep) {
    const mine = rec;
    if (!mine || (keep && mine.ending)) return;
    if (keep && mine.startedAt) {
      mine.ending = true;
      clearInterval(mine.timer);
      if (mine.recorder.state !== 'inactive') mine.recorder.stop(); // `dataavailable`, then `stop`
    } else {
      rec = null; // finishRecording drops a recording that is not the current one
      release(mine);
    }
    updateComposer();
  }

  // The recorder stopped: on request, or by itself when the microphone went away.
  function finishRecording(mine) {
    release(mine);
    if (rec !== mine) return;
    rec = null;
    const audio = new Blob(mine.chunks, { type: mine.recorder.mimeType || '' });
    updateComposer();
    if (Date.now() - mine.startedAt < MIN_RECORD_MS || !audio.size) return; // as if cancelled
    if (audio.size > MAX_AUDIO_BYTES) showError(VOICE_REFUSALS[413]);
    else runTurn(null, audio);
  }

  // --- reading replies aloud

  function setSpeaking(on) {
    speakChosen = true;
    els.speak.checked = on;
    remember(on ? '1' : '0', SPEAK_KEY);
    if (!on) stopSpeaking();
  }

  // Pieces of at most SPEECH_LIMIT characters, cut after a sentence (. ? ! or a line break) where there is one.
  function speechPieces(text) {
    const pieces = [];
    let piece = '';
    for (const sentence of text.match(/[\s\S]+?(?:[.?!]+(?=\s|$)|\n|$)\s*/g) || []) {
      let rest = [...sentence]; // code points, as the server counts
      if (piece && [...piece].length + rest.length > SPEECH_LIMIT) {
        pieces.push(piece);
        piece = '';
      }
      for (; rest.length > SPEECH_LIMIT; rest = rest.slice(SPEECH_LIMIT)) {
        pieces.push(rest.slice(0, SPEECH_LIMIT).join('')); // one endless sentence
      }
      piece += rest.join('');
    }
    pieces.push(piece);
    return pieces.map((p) => p.trim()).filter(Boolean);
  }

  // Agent messages on the page that `before` (an earlier copy of `displayed`) did not have.
  function unseenReplies(before) {
    const seen = before.filter((m) => m.role === 'agent').map((m) => m.text);
    const unseen = [];
    for (const m of displayed) {
      if (m.role !== 'agent') continue;
      const at = seen.indexOf(m.text);
      if (at === -1) unseen.push(m.text);
      else seen.splice(at, 1);
    }
    return unseen;
  }

  function speak(text) {
    if (!voiceOn || !els.speak.checked || !sessionId) return;
    speech.queue.push(...speechPieces(text));
    pump();
  }

  async function pump() {
    if (speech.busy) return;
    speech.busy = true;
    const run = speech.run;
    while (speech.queue.length && run === speech.run) {
      els.speakStop.hidden = false;
      await sayPiece(speech.queue.shift(), run);
    }
    speech.busy = false;
    if (speech.queue.length) pump(); // stopped, and asked again before this loop noticed
    else hideButton(els.speakStop, els.speak);
  }

  async function sayPiece(text, run) {
    speech.abort = new AbortController();
    try {
      const response = await fetch(`${sessionUrl(sessionId)}/speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: speech.abort.signal,
      });
      if (response.status === 503) setVoice(false);
      if (response.status !== 200) return; // 204: nothing in it can be pronounced
      const sound = await response.blob();
      if (run === speech.run) await play(sound);
    } catch {
      // stopped, or the server cannot be reached: the text is on the page anyway
    }
  }

  // Resolves when the sound ended, failed or was stopped. Needs `media-src blob:` in the CSP.
  function play(sound) {
    return new Promise((resolve) => {
      const audio = speech.audio || (speech.audio = new Audio());
      const url = URL.createObjectURL(sound);
      const finish = () => {
        if (speech.finish !== finish) return;
        speech.finish = null;
        audio.onended = null;
        audio.onerror = null;
        URL.revokeObjectURL(url);
        resolve();
      };
      speech.finish = finish;
      audio.onended = finish;
      audio.onerror = finish;
      audio.src = url;
      audio.play().catch((error) => {
        finish();
        if (!error || error.name !== 'NotAllowedError') return;
        stopSpeaking(); // the browser wants a click first: the rest of the queue would fail too
        if (els.error.hidden) showError(NO_AUTOPLAY);
      });
    });
  }

  function stopSpeaking() {
    speech.run += 1;
    speech.queue.length = 0;
    if (speech.abort) speech.abort.abort();
    if (speech.audio) speech.audio.pause();
    if (speech.finish) speech.finish();
  }

  function setVoice(on) {
    voiceOn = on;
    els.voiceBar.hidden = !on;
    els.mic.hidden = !on;
    if (on) return;
    stopRecording(false);
    stopSpeaking();
  }

  async function loadVoice() {
    try {
      const response = await fetch('/healthz', { cache: 'no-store' });
      if (!response.ok || (await response.json()).voice !== true) return;
    } catch {
      return; // text chat only
    }
    const saved = recall(SPEAK_KEY);
    speakChosen = saved !== null;
    els.speak.checked = saved === '1';
    setVoice(true);
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
  els.mic.addEventListener('click', () => (rec ? stopRecording(Boolean(rec.startedAt)) : startRecording()));
  els.recCancel.addEventListener('click', () => stopRecording(false));
  els.speak.addEventListener('change', () => setSpeaking(els.speak.checked));
  els.speakStop.addEventListener('click', stopSpeaking);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !rec) return;
    event.preventDefault();
    stopRecording(false);
  });

  async function init() {
    loadHints();
    loadVoice();
    const saved = recall();
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
