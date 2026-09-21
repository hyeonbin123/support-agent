# 음성 채널 (5단계)

상담 에이전트는 글로 된 대화를 기준으로 만들고 쟀다. 콜센터에서는 고객의 말이 음성 인식을 거쳐 들어온다. 이 단계는 그 사이에 무엇이 깨지는지를 잰다: 고객이 하려던 말(글) → 말로 풀어 쓰기 → 음성 합성 → 음성 인식 → 에이전트.

측정 규칙과 결과는 [experiments.md](experiments.md)의 "5단계"에 있다.

## 설치와 실행

음성 모델은 선택 의존성이다 (GPU용 torch를 포함해 수 GB). 테스트와 나머지 기능은 이것 없이 돈다.

```bash
uv sync --group voice
uv run --group voice python -m support_agent.run --tasks dev --trials 4 --voice V1
uv run python -m support_agent.analyze voice reports/<텍스트 실행> reports/<음성 실행>
uv run python -m support_agent.analyze voice-worst reports/<개발용 음성 실행>
```

- `--tts-device cpu`, `--stt-device cpu`로 모델을 CPU에 둘 수 있다 (GPU 메모리가 모자랄 때)
- 기록(`episodes.jsonl`)의 `voice` 항목에 발화마다 고객이 쓴 글(`said`), 말로 푼 글(`spoken`), 인식된 글(`heard`), 에이전트가 받은 글(`text`), 음성 길이와 합성·인식 시간이 남는다. 음성 파일은 남기지 않는다

## 채팅 서비스에서 말로 상담하기

`SUPPORT_AGENT_VOICE=1`로 서비스를 띄우면 채팅 화면에 마이크 버튼이 생긴다. 음성 모델이 필요하므로 Docker가 아니라 호스트에서 띄운다.

```bash
uv sync --group voice
export SUPPORT_AGENT_VOICE=1 SUPPORT_AGENT_ADMIN_TOKEN=local-admin    # PowerShell: $env:SUPPORT_AGENT_VOICE = "1"
uv run --group voice uvicorn support_agent.service.app:create_app --factory --port 8062
```

- `POST /api/sessions/{id}/voice`: 본문은 브라우저가 녹음한 음성 그대로(webm, ogg, wav, mp4. 5 MB까지). 인식된 문장을 `heard` 이벤트로 먼저 보내고, 그 뒤는 글로 보낸 메시지와 같은 길을 간다. 인식된 것이 없으면 턴을 쓰지 않고 다시 말해 달라고 한다
- `POST /api/sessions/{id}/speech`: 글을 말로 풀어 쓴 뒤 합성한 WAV. 에이전트의 답을 읽어 주는 데 쓴다 (600자까지). 세션에 묶어서 아무나 쓰는 음성 합성 서비스가 되지 않게 했다
- 음성으로 들어온 메시지는 감사 로그에 인식 원문과 함께 남는다 (`customer_message.via`)
- 서비스의 인식기는 말이 없는 구간을 먼저 걸러 낸다 (faster-whisper의 VAD). 실제 마이크로 해 보니, 말없이 버튼만 눌렀다 뗀 녹음을 Whisper가 "다음 영상에서 만나요."로 적었다. 영상 자막에서 배운 상투 문장이다. 측정용 채널은 합성 음성이라 무음이 없으므로 이 옵션을 끈 채로 둔다 (규칙에 적은 설정 그대로)
- 음성을 켜면 응답 헤더가 두 군데 달라진다: `Permissions-Policy`의 `microphone=(self)`, CSP의 `media-src 'self' blob:`
- `SUPPORT_AGENT_TTS_DEVICE`, `SUPPORT_AGENT_STT_DEVICE`로 모델을 CPU에 둘 수 있다

## 구성

| 파일 | 하는 일 |
|---|---|
| `voice/verbalize.py` | 쓴 글을 사람이 말하는 대로 풀어 쓴다. 모델 없는 순수 함수 |
| `voice/speech.py` | MeloTTS(합성)와 faster-whisper(인식) 래퍼. 라이브러리는 생성자 안에서만 불러온다 |
| `voice/channel.py` | 풀어 쓰기 → 합성 → 인식 → (정규화). 같은 입력의 결과는 캐시한다 |
| `voice/metrics.py` | 글자 오류율, 엔티티 생존율, 본인 확인 성공률과 짝지은 부트스트랩 구간 |
| `service/voice_frontend.py` | 서비스 앞단: 녹음을 글로, 답을 소리로. 모델 호출은 한 번에 하나 |

## 말로 풀어 쓰기

음성 합성기에 "O-91010"이나 "38,900원"을 그대로 주면 아무도 그렇게 말하지 않는 소리가 나온다. 그래서 합성 전에 한글로만 된 글로 바꾼다.

| 쓴 글 | 말로 푼 글 |
|---|---|
| `010-0000-9103` | 공일공, 공공공공, 구일공삼 (한 자리씩) |
| `O-91010`, `RT-O-91009-1` | 오 다시 구일공일공, 알 티 다시 오 다시 구일공공구 다시 일 |
| `38,900원`, `10,000원` | 삼만 팔천구백 원, 만 원 (한자어 수, "일만"이 아니라 "만") |
| `9월 14일`, `6월`, `10월` | 구월 십사일, 유월, 시월 |
| `2개`, `20개`, `10시` | 두 개, 스무 개, 열 시 (세는 말 앞에서는 고유어 수) |
| `7일`, `3개월`, `5층` | 칠 일, 삼 개월, 오 층 |
| `hajun@example.com` | 에이치 에이 제이 유 엔 골뱅이 이그잼플 닷컴 |
| `250mm`, `10%` | 이백오십 밀리미터, 십 퍼센트 |

개발용 기준선 기록에 있는 고객 발화 512개가 모두 한글, 공백, 문장부호만 남는지를 테스트가 확인한다.

## 정한 것과 이유

- **음성 모델은 옆 프로젝트에서 이미 잰 것을 쓴다.** 한국어 합성은 MeloTTS(재인식 CER 5.5%로 사람 녹음 5.4%와 비슷했다), 인식은 faster-whisper `large-v3-turbo`다. MeloTTS는 transformers 4.27을 고정하고 Windows에서 우회 세 개가 필요하다 (`speech.py`의 `_prepare_melo_on_windows`). 본체는 transformers를 쓰지 않아 한 환경에 같이 설치된다. MeloTTS의 시연 화면만 쓰는 gradio는 설치에서 뺐다
- **고객 → 에이전트 방향만 바꾼다.** 시뮬레이터는 자기가 쓴 글을 기억하고, 에이전트의 답은 글 그대로 시뮬레이터에 간다. 그래서 텍스트 실행과의 차이는 "에이전트가 잘못 들었다"에서만 나온다
- **같은 입력이면 같은 소리.** 합성의 표본 추출 잡음을 발화마다 정해진 seed로 고정하고, 인식은 temperature 0만 쓴다. 그래서 왕복 결과를 캐시해도 측정값이 달라지지 않는다. 캐시 키에는 말로 푼 글과 seed 말고도 모델 이름, 장치, 옵션, 패키지 버전(melotts, torch, faster-whisper, ctranslate2)이 들어간다. 버전이 바뀌면 예전 결과를 쓰지 않는다
- **음성 파일을 커밋하지 않는다.** 글과 seed로 다시 만들 수 있다
