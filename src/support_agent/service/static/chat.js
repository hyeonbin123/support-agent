'use strict';

// [파일 역할] 고객 채팅 화면(index.html)의 동작 전부: 세션 시작·복원, 메시지 전송과 SSE 응답 읽기, 진행 표시,
// 승인 대기 칩, 주기적 동기화, 그리고 선택 기능인 음성(녹음해서 보내기, 답을 소리로 듣기).
// [흐름] init → (저장된 세션이 있으면 refresh로 복원, 없으면 startSession) → 사용자가 보내면 runTurn이 POST
// 응답을 SSE로 읽어 onEvent로 화면을 바꿈 → 5초마다 poll이 서버 대화와 맞춤(승인 결과 문장이 이렇게
// 들어온다).
// [설계 이유] 서버는 턴이 끝날 때만 대화를 저장하므로, 화면에 먼저 그린 말풍선과 서버 목록이 잠시 어긋날 수
// 있다. reconcile이 둘을 비교해 다를 때만 다시 그린다. 서버의 글은 모두 textContent로 넣는다(innerHTML 없음).
// [연결] app.py의 /api/sessions, /messages, /voice, /speech, /healthz와 /static/demo.json을 부른다.

(() => {
  // LIMIT: 메시지 글자 수 한도(서버 기본값 max_message_chars와 같음). 서버처럼 코드 포인트 단위로 센다.
  // COUNTER_FROM: 이 글자 수부터 남은 글자 표시를 보인다. POLL_MS: 대화를 다시 읽는 주기(5초).
  const LIMIT = 1000; // characters, counted as code points like the server does
  const COUNTER_FROM = 800;
  const POLL_MS = 5000;
  // 세션 번호를 sessionStorage에 두는 키. 같은 탭에서 새로 고침해도 대화가 이어지고, 탭을 닫으면 사라진다.
  // 세션 번호가 곧 열쇠이므로 오래 남기지 않는다.
  const STORAGE_KEY = 'support-agent.session-id';
  const THINKING = '답변 작성 중…';
  const ROLE_NAMES = { customer: '고객', agent: '상담원' };
  // 세션이 닫힌 이유(서버의 세션 상태)별 종료 안내.
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

  // 화면 상태. epoch는 새 상담을 시작할 때마다 늘어, 예전 세션에 대한 늦은 응답을 버리는 데 쓴다.
  // recoverable은 지금 보이는 오류가 저절로 사라져도 되는 종류인지 표시한다. displayed는 그려 둔 말풍선
  // 목록으로 서버 대화와 비교하는 기준이다.
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

  // sessionStorage 쓰기·읽기. 저장소가 막혀 있어도 예외로 멈추지 않고 이 페이지를 보는 동안은 동작한다.
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

  // 말풍선 하나를 붙인다. 글은 textContent로만 넣고, 화면 낭독기를 위해 "고객: "/"상담원: "을 숨긴 글로 앞에
  // 둔다.
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

  // 말풍선을 뒤에서부터 지워 length개만 남긴다(보내지 못한 메시지를 되돌리거나 전부 다시 그릴 때).
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

  // 승인 대기 칩을 다시 그린다. 고객에게는 승인 번호와 환불 예정 금액을 보인다.
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

  // 입력창·보내기·마이크 버튼의 사용 가능 여부를 정한다. 턴이 도는 중이거나, 세션이 닫혔거나, 녹음 중이면
  // 입력을 막는다. 앞뒤 공백을 뺀 길이가 0이거나 한도를 넘으면 보내기를 막는다.
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

  // 서버가 준 세션 상태(대화, 대기 중인 승인, 세션 상태)를 화면에 반영한다.
  function applyTranscript(transcript) {
    reconcile(transcript.messages || []);
    setPending(transcript.pending_approvals);
    status = transcript.status;
    renderStatus();
  }

  // ------------------------------------------------------------ sessions

  // 새 상담을 시작한다. 진행 중인 요청·녹음·재생을 모두 멈추고 화면을 비운 뒤 POST /api/sessions로 새 세션을
  // 만든다. 응답이 오기 전에 또 새 상담을 눌렀으면(epoch가 바뀜) 늦은 응답은 버린다.
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
  // 서버의 세션 상태를 다시 읽어 화면을 맞춘다. 세션이 없어졌으면(404) 새 상담을 열고 그렇게 했다고 알린다.
  // 승인 대기였던 건이 목록에서 사라졌다면 담당자가 결정한 것이고, 그 결과 문장은 새로 생긴 답이므로 음성
  // 듣기를 켠 고객에게 읽어 준다.
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

  // 주기적 동기화. 턴이 도는 중, 녹음 중, 이전 동기화가 안 끝남, 탭이 숨겨짐이면 건너뛴다. 세션이 닫혔어도
  // 승인 대기가 남아 있으면 결정 결과를 받으려고 계속 읽는다.
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

  // SSE 프레임 하나(빈 줄로 구분된 덩어리)를 {event, data}로 바꾼다. data 줄이 여러 개면 줄바꿈으로 이어
  // JSON으로 읽고, 읽을 수 없는 프레임은 버린다.
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

  // fetch 응답 본문을 스트림으로 읽어 SSE 프레임 단위로 넘긴다. EventSource는 GET만 되므로 POST로 보낸
  // 메시지의 응답을 직접 읽는다.
  // TextDecoder의 stream 옵션으로 조각 경계에서 잘린 한글(UTF-8 여러 바이트)이 깨지지 않게 하고, CRLF를 LF로
  // 맞춘 뒤 빈 줄(\n\n)마다 자른다.
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
  // 서버 이벤트별 화면 반영: status·tool_result는 "답변 작성 중", tool은 "주문 조회 중" 같은 진행 표시,
  // approval은 승인 대기 칩, error는 오류 줄, reply는 상담원 말풍선(음성 듣기를 켰으면 읽어 줌), end는 세션
  // 상태 갱신. end면 true를 돌려준다.
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

  // 서버가 메시지를 받지 않았을 때(HTTP 오류) 고객에게 보일 문장. 음성 메시지면 음성용 문장을 먼저 고른다.
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
  // 한 턴을 보낸다. 글 메시지는 고객 말풍선을 바로 그리고, 녹음은 서버가 heard 이벤트로 인식 결과를 보내야
  // 말풍선을 그린다.
  // 응답이 끝 이벤트(end) 없이 끊기면 오류로 본다. 이미 서버가 메시지를 받았다면(accepted) 턴은 서버에서
  // 끝까지 가므로, 주기적 동기화가 답을 가져온다고 안내한다.
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
      // 녹음이면 /voice에 녹음 바이트를 그대로(브라우저가 정한 형식으로), 글이면 /messages에 JSON으로 보낸다.
      // controller로 새 상담 시작 시 요청을 끊을 수 있다.
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
        // 서버가 음성을 쓸 수 없다고 하면(503) 음성 막대와 마이크 버튼을 숨긴다.
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
      // 서버가 받지 않았거나 답 없이 끝났으면 먼저 그린 고객 말풍선을 지우고, 서버 목록을 다시 읽어 맞춘다.
      if (!accepted || (ended && !replied)) truncate(index);
      await refresh(); // busy stays true meanwhile, so a poll cannot interleave
      if (mine !== epoch) return;
      busy = false;
    }
    // Give the text back when the server did not record it (refused, busy, or never reached).
    // 서버에 기록되지 않은 글이면 입력창에 되돌려 놓아 다시 쓰지 않아도 되게 한다.
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

  // 녹음 형식 후보(브라우저가 지원하는 첫 번째를 쓴다). 서버의 인식기는 webm·ogg·mp4 등을 그대로
  // 받는다(docs/voice.md).
  const MIC_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  // 녹음 한 번은 30초까지(서버는 풀었을 때 60초까지 받는다). 0.3초보다 짧은 녹음은 잘못 누른 것으로 보고
  // 보내지 않는다. 5 MB(서버 기본 한도)를 넘는 녹음은 보내기 전에 거절한다. /speech 한 번에는 600자까지(서버
  // max_tts_chars).
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
  // 음성 끝점의 HTTP 오류별 안내: 413 너무 긺, 422 빈 녹음, 503 음성 기능 꺼짐.
  const VOICE_REFUSALS = {
    413: '녹음이 너무 깁니다. 짧게 나누어 다시 말씀해 주세요.',
    422: '녹음된 소리가 없습니다. 다시 말씀해 주세요.',
    503: '지금은 음성 기능을 사용할 수 없습니다. 메시지를 글로 입력해 주세요.',
  };

  let voiceOn = false;
  let speakChosen = false; // the listening toggle was set, by the customer or by the first voice message
  // 답 읽어 주기의 상태: 읽을 조각 큐, 멈출 때마다 늘어나는 run 번호(이전 재생 루프를 끝내는 데 씀), 재생 중
  // 여부, 재사용하는 Audio 객체, 진행 중인 요청을 끊을 AbortController, 지금 재생의 마무리 함수.
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

  // 마이크 버튼과 녹음 상태 표시를 그린다. 브라우저가 마이크 권한을 묻는 동안(startedAt 없음)은 "마이크 연결
  // 중…"으로 보이고, 녹음이 시작되면 버튼이 "눌러서 전송"으로 바뀐다.
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

  // 0.25초마다 경과 시간을 보이고, 30초가 되면 녹음을 끝내 보낸다.
  function tick() {
    if (!rec || !rec.startedAt || rec.ending) return;
    const elapsed = Date.now() - rec.startedAt;
    if (elapsed >= MAX_RECORD_MS) stopRecording(true);
    else setText(els.recStatus, `녹음 중 ${Math.floor(elapsed / 1000)}초 / ${MAX_RECORD_MS / 1000}초`);
  }

  // 녹음기와 마이크 트랙을 멈춘다. 트랙을 멈춰야 브라우저의 마이크 사용 표시가 꺼진다.
  function release(take) {
    clearInterval(take.timer);
    if (take.recorder && take.recorder.state !== 'inactive') take.recorder.stop();
    if (take.stream) for (const track of take.stream.getTracks()) track.stop(); // the browser's mic sign goes off
  }

  // 녹음을 시작한다. 답을 읽어 주는 중이면 그 소리가 녹음되지 않게 먼저 멈춘다. 마이크나 MediaRecorder가
  // 없거나 권한을 거절하면 안내를 보인다. 권한을 묻는 사이 취소했으면 얻은 마이크를 바로 놓는다.
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
  // keep이 true면 녹음기를 멈춰 dataavailable → stop 순으로 finishRecording이 보내게 하고, false면 녹음을
  // 버린다.
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
  // 녹음기가 멈춘 뒤 조각을 하나의 Blob으로 모아 runTurn으로 보낸다. 지금 녹음이 아니면(취소됨) 버리고, 너무
  // 짧거나 비었으면 취소처럼 아무것도 보내지 않는다.
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

  // "답변을 음성으로 듣기" 선택을 저장한다(같은 탭 안에서 기억). 끄면 지금 읽는 것도 멈춘다.
  function setSpeaking(on) {
    speakChosen = true;
    els.speak.checked = on;
    remember(on ? '1' : '0', SPEAK_KEY);
    if (!on) stopSpeaking();
  }

  // Pieces of at most SPEECH_LIMIT characters, cut after a sentence (. ? ! or a line break) where there is one.
  // 긴 답을 /speech 한도(600자) 안의 조각으로 나눈다. 가능하면 문장 끝(마침표·물음표·느낌표 또는 줄바꿈)
  // 뒤에서 자르고, 끝없이 긴 문장 하나는 600자씩 자른다. 서버처럼 코드 포인트로 센다.
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
  // before에는 없던 상담원 답을 고른다. 같은 글의 답이 여러 번 있을 수 있어 하나씩 지워 가며 센다.
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

  // 음성 모드이고 듣기를 켰을 때만 답을 조각내 큐에 넣고 재생 루프를 돌린다.
  function speak(text) {
    if (!voiceOn || !els.speak.checked || !sessionId) return;
    speech.queue.push(...speechPieces(text));
    pump();
  }

  // 큐의 조각을 하나씩 받아 재생한다. 그 사이 멈춤(run 번호가 바뀜)이 있으면 루프를 끝낸다.
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

  // 조각 하나를 서버에서 WAV로 받아 재생한다. 서버는 이 세션의 상담원 답에 있는 글만 합성하고(app.py speak),
  // 204면 소리로 낼 것이 없다는 뜻이다.
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
  // Blob을 blob: 주소로 재생하고 끝나면 주소를 해제한다. 브라우저가 사용자 동작 전 자동 재생을
  // 막으면(NotAllowedError) 나머지 큐도 실패하므로 읽기를 멈추고 한 번 눌러 달라고 안내한다.
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

  // 읽기를 모두 멈춘다: run 번호를 올려 루프를 끝내고, 큐를 비우고, 받는 중인 요청과 재생 중인 소리를 끊는다.
  function stopSpeaking() {
    speech.run += 1;
    speech.queue.length = 0;
    if (speech.abort) speech.abort.abort();
    if (speech.audio) speech.audio.pause();
    if (speech.finish) speech.finish();
  }

  // 음성 막대와 마이크 버튼을 보이거나 숨긴다. 끌 때는 녹음과 읽기도 멈춘다.
  function setVoice(on) {
    voiceOn = on;
    els.voiceBar.hidden = !on;
    els.mic.hidden = !on;
    if (on) return;
    stopRecording(false);
    stopSpeaking();
  }

  // /healthz가 voice: true일 때만 음성 기능을 켠다. 듣기 선택은 이전에 저장한 값이 있으면 그대로, 없으면 첫
  // 음성 메시지 때 켠다(runTurn의 heard 처리).
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

  // 체험 안내에 시험해 볼 가상 고객(이름 · 전화번호 — 시나리오)을 /static/demo.json에서 읽어 보인다. 파일이
  // 없거나 읽지 못하면 그냥 넘어간다.
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

  // --- 이벤트 연결 ---
  // 폼 제출은 페이지를 다시 읽지 않고 send로 처리한다.
  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    send();
  });
  els.input.addEventListener('input', updateComposer);
  els.input.addEventListener('keydown', (event) => {
    // Enter는 보내기, Shift+Enter는 줄바꿈. 한글 조합을 확정하는 Enter(isComposing, keyCode 229)로는 보내지
    // 않는다.
    // isComposing: Enter that only confirms a Hangul composition must not send
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    send();
  });
  $('new-session').addEventListener('click', () => startSession({ focus: true }));
  $('notice-new').addEventListener('click', () => startSession({ focus: true }));
  // 마이크 버튼은 누를 때마다 녹음 시작 ↔ 녹음 끝내고 보내기. 권한을 묻는 중(startedAt 없음)에 누르면 취소다.
  // Esc도 녹음을 취소한다.
  els.mic.addEventListener('click', () => (rec ? stopRecording(Boolean(rec.startedAt)) : startRecording()));
  els.recCancel.addEventListener('click', () => stopRecording(false));
  els.speak.addEventListener('change', () => setSpeaking(els.speak.checked));
  els.speakStop.addEventListener('click', stopSpeaking);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !rec) return;
    event.preventDefault();
    stopRecording(false);
  });

  // 시작: 가상 고객 안내와 음성 기능 여부를 따로 읽고, 저장된 세션이 있으면 서버에서 복원(실패하면 자동
  // 재시도 안내), 없으면 새 세션을 만든다. 그 뒤 5초마다 동기화한다.
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
