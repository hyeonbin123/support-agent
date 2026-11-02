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

### 음성 모델의 리비전 (2026-10-29 보탬)

음성 모델 세 가지는 처음 쓸 때 Hugging Face 허브에서 받아 캐시(`HF_HOME/hub`, `HF_HOME`의 기본값은 `~/.cache/huggingface`)에 둔다. 코드는 리비전을 고정하지 않고 저장소의 `main`을 받는다. 이 PC의 캐시에는 저장소마다 아래 스냅숏(커밋) 하나만 있고 2026-09-13부터 있었다 (첫 음성 측정 2026-09-21보다 앞). 그래서 기록된 음성 측정(5·9·11단계)은 이 파일들로 돈 것으로 본다. 그 기록들의 manifest에는 리비전이 남지 않았다. 2026-10-29에 본 허브의 `main`도 세 저장소 모두 같은 커밋이다.

| 쓰임 | 허브 저장소 | 스냅숏(커밋) |
|---|---|---|
| 합성: MeloTTS 한국어 설정·체크포인트 | `myshell-ai/MeloTTS-Korean` | `0207e5adfc90129a51b6b03d89be6d84360ed323` |
| 합성의 한국어 앞단: 토크나이저·BERT (melo가 고정 id로 부른다) | `kykim/bert-kor-base` | `1779cc0982ada0216dd6de0dd4e86fb78201926d` |
| 인식: `large-v3-turbo` (CTranslate2 변환본) | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` |

- 2026-10-29부터 음성 실행의 `manifest.json`에 `voice_extra`가 남는다 (11단계 `voice_real run`도 같다): `versions`(MeloTTS가 고정하는 `transformers`·`librosa`, 그 transformers가 범위를 묶는 `tokenizers`·`huggingface-hub`, 그리고 `numpy`의 버전)와 `model_revisions`(위 세 저장소마다, 모델을 올린 직후 캐시의 `main`이 가리키는 스냅숏 이름. 찾지 못하면 `?`). 파일 sha256은 남기지 않는다 (가중치가 수백 MB~1.6 GB라 실행마다 HDD에서 수십 초가 든다)
- `voice_extra`는 왕복 캐시 키와 짝짓기 검사(`analyze same-setup`, 끼워 넣기 검사, 11단계 재현 확인)에 넣지 않는다. 넣으면 이 항목이 없는 기존 음성 기록 전부와 "다름"이 되기 때문이다. 두 실행의 이 값은 사람이 대조한다. 끼워 넣은 기록(`analyze stitch`)은 원래 실행의 manifest를 그대로 쓰므로, 다시 잰 에피소드 쪽 값은 `stitched.rerun`이 가리키는 실행의 manifest에 있다
- 한 번 받은 뒤에는 `HF_HUB_OFFLINE=1`로 띄우면 허브에 묻지 않는다. 캐시의 `refs/main`이 가리키는 스냅숏을 그대로 쓰고, 캐시에 없는 파일은 받지 않고 바로 오류를 낸다. transformers 4.27.4는 이 변수를 직접 읽지 않지만 파일을 huggingface_hub(0.36.2)로 받으므로 BERT도 캐시에서 온다 (2026-10-29 CPU로 확인: melo 불러오기(토크나이저), MeloTTS 설정·체크포인트, BERT 가중치, 인식 모델 스냅숏이 모두 캐시에서 풀렸고, 캐시에 없는 파일은 곧바로 `LocalEntryNotFoundError`). 처음 한 번은 온라인으로 받아야 하고, BERT 가중치는 첫 합성 때 받는다

```bash
export HF_HUB_OFFLINE=1    # PowerShell: $env:HF_HUB_OFFLINE = "1"
uv run --group voice python -m support_agent.run --tasks dev --trials 4 --voice V1
```

## 채팅 서비스에서 말로 상담하기

`SUPPORT_AGENT_VOICE=1`로 서비스를 띄우면 채팅 화면에 마이크 버튼이 생긴다. 음성 모델이 필요하므로 Docker가 아니라 호스트에서 띄운다.

```bash
uv sync --group voice
export SUPPORT_AGENT_VOICE=1 SUPPORT_AGENT_ADMIN_TOKEN=local-admin    # PowerShell: $env:SUPPORT_AGENT_VOICE = "1"
uv run --group voice uvicorn support_agent.service.app:create_app --factory --port 8062 --no-access-log
```

- `POST /api/sessions/{id}/voice`: 본문은 브라우저가 녹음한 음성 그대로(webm, ogg, wav, mp4. 5 MB까지). 인식된 문장을 `heard` 이벤트로 먼저 보내고, 그 뒤는 글로 보낸 메시지와 같은 길을 간다. 인식된 것이 없으면 턴을 쓰지 않고 다시 말해 달라고 한다. 동시에 도는 턴이 한도(4)에 닿으면 글 메시지와 같이 503과 `Retry-After`를 받는다. 화면은 이 503과 음성 기능이 꺼진 503(`Retry-After` 없음)을 가려, 앞의 것에는 잠시 뒤 다시 보내 달라고만 안내하고 마이크 버튼을 그대로 둔다 (2026-10-24 고침. 전에는 둘 다 음성 막대를 숨겨 붐빌 때 음성 모드가 꺼졌다)
- `POST /api/sessions/{id}/speech`: 글을 말로 풀어 쓴 뒤 합성한 WAV. 에이전트의 답을 읽어 주는 데 쓴다 (600자까지). 세션에 묶어서 아무나 쓰는 음성 합성 서비스가 되지 않게 했다. 동시에 2개까지 합성하고(설정 `SUPPORT_AGENT_MAX_CONCURRENT_SPEECH`, 2026-10-24부터) 넘으면 기다리지 않고 429와 `Retry-After: 2`로 돌려보낸다. 화면은 그 시간 뒤 한 번만 다시 청하고, 그래도 429면 그 조각은 건너뛴다 (글은 화면에 있다). 전에는 한도가 없어 겹친 요청이 모두 작업 스레드를 쥔 채 합성 잠금(모델 호출은 한 번에 하나)을 기다렸다 ([service.md](service.md) "한도")
- 음성으로 들어온 메시지는 감사 로그에 인식 원문과 함께 남는다 (`customer_message.via`)
- 에이전트는 인식 원문이 아니라 고친 글을 받는다: 측정에서 효과가 확인된 표기 규칙(V2, `voice/normalize.py`)을 먼저 적용하고, 그다음 사람이 부른 번호를 숫자로 되돌린다. 인식 원문은 감사 로그에 남는다
- 사람이 부른 번호를 숫자로 되돌린다 (`voice/spoken.py`). 실제 마이크로 해 보니 전화번호가 "공일공 공공공공 육일팔구입니다"처럼 한글로 적혔다. 합성 음성에서는 `010-0000-9103`으로 적히던 것이다. 휴대전화 번호(010과 여덟 자리)와 주문 번호("오 다시"와 다섯 자리)만 되돌리고, 뒤에 붙은 조사 "이고"를 숫자 2로 읽지 않도록 자릿수를 정확히 센다. 이 변환은 측정한 V2 정규화와 별개다 (V2는 개발용 기록에서, 이것은 실제 마이크 시험에서 나왔다)
- 서비스의 인식기는 말이 없는 구간을 먼저 걸러 낸다 (faster-whisper의 VAD). 실제 마이크로 해 보니, 말없이 버튼만 눌렀다 뗀 녹음을 Whisper가 "다음 영상에서 만나요."로 적었다. 영상 자막에서 배운 상투 문장이다. 측정용 채널은 합성 음성이라 무음이 없으므로 이 옵션을 끈 채로 둔다 (규칙에 적은 설정 그대로)
- 음성을 켜면 응답 헤더가 두 군데 달라진다: `Permissions-Policy`의 `microphone=(self)`, CSP의 `media-src 'self' blob:`
- `SUPPORT_AGENT_TTS_DEVICE`, `SUPPORT_AGENT_STT_DEVICE`로 모델을 CPU에 둘 수 있다

## V4: 발신 번호로 하는 본인 확인 (9단계)

`--voice V4`는 V2와 같은 채널에 본인 확인 방식 하나를 바꾼다. 규칙과 결과는 [experiments.md](experiments.md)의 "9단계".

9단계의 처음 규칙(이름의 자모 1개 허용)은 재기 전에 정한 관문에서 멈췄다: 자모 1개 허용이 가상 고객 109명 안에서 다른 사람의 이름 6쌍을 받아들인다. 2026-10-04 이름을 정확히 맞추는 규칙으로 다시 적었고(같은 관문 0건), 에이전트는 서비스의 기본 모델 `qwen3.5:4b`로 잰다. 개발용 24과제 × 4회에서 본인 확인 성공이 V2 52.1%에서 V4 80.2%로(+28.1%p [+12.5, +44.8]), pass^1이 35.4%에서 52.1%로 올랐다. 설계 변경의 효과이고(인식이 나아진 것이 아니다), 시험용은 다른 날(2026-10-09~10) 쟀다. 판정: 발신 번호 확인이 본인 확인을 되찾는다 (시험용 40과제 × 4회 V2 43.8% → V4 58.1%, +14.4%p [+6.2, +23.1]). 발신 번호 확인이 pass^1을 되찾는다 (24.4% → 33.1%, +8.8%p [+1.2, +16.9]). 첫 두 시험용 시도는 V2가 첫 에피소드에서 멈춰 그만뒀고, 다른 프로그램의 GPU 사용이 5분 동안 낮을 때 시작하는 규칙을 더해 다시 쟀다. 개발용 실행에서 정답에 없는 쓰기도 7 → 22건으로 늘었다 (규정을 프롬프트에만 둔 P0로 쟀다. 그중 9건은 서비스 구성(P1)이었다면 도구가 거절했을 규정 위반이고, 13건은 P1에서도 막히지 않는 쓰기다). 시험용에서는 16 → 36건이고 그중 통과된 규정 위반은 4 → 7건이다. 2026-10-14: 이 단락의 수치는 말로 풀어 쓰기가 송장 번호를 범위로 읽던 것을 고친 뒤, 그 발화가 든 에피소드(개발용 6, 시험용 1)만 다시 재어 원래 기록에 끼워 넣은 값이다. 수정 전 코드로 낸 값은 개발용 본인 확인 53.1% → 80.2%, pass^1 35.4% → 51.0%, 시험용 본인 확인 44.4% → 58.1% (+13.8%p [+6.2, +21.9]), pass^1 25.0% → 33.1% (+8.1%p [+0.0, +16.9], "확인하지 못함")였다. pass^1 문장은 다시 잰 시험용 V2 에피소드 하나가 성공에서 실패로 간 몫으로 바뀌었다 ([experiments.md](experiments.md) 9단계 끝). (2026-11-03 덧붙임) pass^1 판정의 견고성 경고: 이 판정은 등록한 규칙(과제 단위로 짝지은 백분위 부트스트랩 구간)으로만 0보다 위다. 같은 과제별 pass^1 차이에 과제 단위 정확 부호 뒤집기 순열 검정(양측)을 대면 p = 0.058이고, 차이의 부호만 보는 검정(V4가 높은 과제 13, V2가 높은 과제 7, 같은 과제 20)은 p ≈ 0.26이라 관례적인 5% 기준에서는 차이를 확인하지 못한다. 판정 규칙과 문장은 바꾸지 않았고, 값은 커밋된 기록에서 다시 셌다 (README 9단계 절의 같은 덧붙임).

- 통화가 고객이 가입한 전화번호에서 온다고 보고, 하니스가 그 번호를 도구 문맥에 넣는다 (에이전트에게는 보이지 않는다)
- 에이전트의 도구 목록에서 `find_customer`가 빠지고 `verify_caller(name)`가 들어간다. 번호가 가리키는 고객 한 명의 이름과만 비교하고, 공백과 끝의 '고객님'·'고객'·'님'을 뗀 뒤 글자까지 같아야 한다 (`find_customer`와 같은 기준. "배성분"은 "배성훈"으로 확인되지 않는다). 결과에는 등록된 이름이 온다
- 시스템 프롬프트 규칙 목록 끝에 한 줄이 붙는다 (`prompts/caller_id.md`). V0~V2의 프롬프트와 도구 목록은 그대로다
- 서비스의 음성 모드와 MCP 서버는 바뀌지 않는다 (브라우저 마이크에는 발신 번호가 없다)
- `uv run python -m support_agent.analyze caller-id <실행 ...>`: 확인 도구를 부르지 않은 에피소드, 허용 오차로 통과한 확인(허용 오차 0에서는 0이어야 한다), 없는 도구(`find_customer`) 호출. `analyze name-collisions [--k K]`(기본은 등록한 허용 오차 0): seed 고객의 번호와 다른 고객 이름의 쌍 가운데 허용 오차가 받아들이는 것

## V4-R: 전화 통화 조건에서 받아쓰기 (11단계 A단계)

`voice/degrade.py`는 합성 음성에 전화 통화 조건을 얹는다: 발화의 20%에 먹먹함(500 ms를 1500 Hz 저역 통과), 분홍·갈색 생성 잡음(전화 대역 안에서 SNR 15 dB ±3 dB), 분당 1회의 돌발 소음(−5 ~ +10 dB), 300–3400 Hz·8 kHz·8비트 μ-law(whisper-ko-ft의 `telephone()`과 같은 구현), 20 ms 프레임의 Gilbert–Elliott 손실(평균 2%, 100 ms 묶음, 잃은 프레임은 무음). 값은 τ-Voice (arXiv 2603.13686)의 Realistic이고, 다르게 정한 것과 이유는 [experiments.md](experiments.md)의 "11단계"에 있다. 무작위는 (기본 seed, 과제, 시도, "degrade", 순번)에서 나온다.

A단계 결과 (송장 번호를 고쳐 다시 잰 9단계 V4 개발용 926발화, 2026-10-14 `reports/20261014-033627-v4r-a-dev-respelled`, 판정 없음): 깨끗한 음성 → 열화한 음성으로 글자 오류율 4.23% → 6.34% (+2.11%p [+1.23, +3.36]), 이름 정확 생존 99/165 → 83/165 (−9.7%p [−19.5, −0.5], 위 끝이 0에 가깝다), 주문·접수 번호 생존 81/125 → 74/125 (−5.6%p [−10.2, −2.0]). 깨끗한 쪽은 9단계 기록과 받아쓰기가 99.6% 같다. 에피소드(B단계)는 재지 않았다. 2026-10-08의 첫 측정(929발화, `reports/20261008-145519-v4r-a-dev`)은 그중 8개가 송장 번호를 범위로 읽은 글로 합성된 수정 전 코드로 낸 값이고, 4.40% → 6.39% (+2.00%p [+1.11, +3.25]), 99/165 → 84/165 (−9.1%p [−18.7, +0.0], 위 끝이 0), 80/126 → 74/126 (−4.8%p [−9.2, −1.3])였다. (2026-11-01 보탬) 이 절의 "글자 오류율"은 README 5단계 결과 절에 정의한 채널 글자 오류율과 같은 계산이다: 고객이 쓴 글과 합성·인식·정규화를 거쳐 나온 글을 공백·문장부호를 빼고 비교하므로, 음성 인식기만의 오류율이 아니다. 위 수치는 그대로다.

```bash
uv run --no-sync python -m support_agent.voice_real plan reports/<V4 개발용 실행>          # CPU: 뽑힌 조건만 확인
uv run --no-sync python -m support_agent.voice_real run reports/<V4 개발용 실행> --official --label dev   # GPU
uv run --no-sync python -m support_agent.voice_real report reports/<위 실행>
```

- 9단계 V4 실행의 고객 발화를 같은 말로 푼 글과 같은 합성 seed로 다시 합성하고, 같은 인식기로 그대로 한 번, 열화를 거쳐 한 번 받아 적는다. 에이전트는 없다. 결과는 발화마다 `utterances.jsonl`(두 받아쓰기, 뽑힌 조건)에 남는다
- 공식 실행은 작업 트리가 깨끗하고, 음성 모델과 장치가 원본 실행과 같아야 한다. 왕복은 `outputs/voice-cache/v4r-roundtrips.jsonl`에 캐시한다 (키에 열화의 버전과 값이 들어 있다)

## 구성

| 파일 | 하는 일 |
|---|---|
| `voice/verbalize.py` | 쓴 글을 사람이 말하는 대로 풀어 쓴다. 모델 없는 순수 함수 |
| `voice/speech.py` | MeloTTS(합성)와 faster-whisper(인식) 래퍼. 라이브러리는 생성자 안에서만 불러온다 |
| `voice/channel.py` | 풀어 쓰기 → 합성 → 인식 → (정규화). 같은 입력의 결과는 캐시한다 |
| `voice/normalize.py` | V2: 인식된 글의 표기를 고치는 규칙 (주문 번호의 O, 이메일의 골뱅이와 도메인, 전화번호 구분, 만 단위 금액). 개발용 V1 기록만 보고 만들었다 |
| `voice/spoken.py` | 사람이 읽은 전화번호·주문 번호를 숫자로 (서비스용, 측정과 무관) |
| `voice/metrics.py` | 채널 글자 오류율(고객이 쓴 글 대비 에이전트가 받은 글), 엔티티 생존율, 본인 확인 성공률과 짝지은 부트스트랩 구간 |
| `voice/degrade.py` | 11단계: 전화 통화 조건(먹먹함, 생성 소음, 돌발 소음, 8 kHz μ-law, 프레임 손실). numpy·scipy만 쓴다 |
| `voice_real.py` | 11단계 A단계: 다시 합성 → 깨끗한·열화한 받아쓰기, 뽑힌 조건의 CPU 관문, 표 |
| `service/voice_frontend.py` | 서비스 앞단: 녹음을 글로, 답을 소리로. 모델 호출은 한 번에 하나 |

## 말로 풀어 쓰기

음성 합성기에 "O-91010"이나 "38,900원"을 그대로 주면 아무도 그렇게 말하지 않는 소리가 나온다. 그래서 합성 전에 한글로만 된 글로 바꾼다.

| 쓴 글 | 말로 푼 글 |
|---|---|
| `010-0000-9103` | 공일공, 공공공공, 구일공삼 (한 자리씩) |
| `5551-0914-9121` (송장 번호처럼 숫자만 셋 이상 묶은 번호) | 오오오일, 공구일사, 구일이일 (전화번호처럼 묶음마다 한 자리씩. 2026-10-13 전에는 뒤 두 묶음을 범위로 읽어 "오천오백오십일 구백십사에서 구천백이십일"이었다. 기록된 고객 발화 21개가 이것으로 바뀐다. 2026-10-14 그 발화가 든 9단계 에피소드 7개와 11단계 A단계를 다시 쟀고, 5단계 개발용 V2의 1개는 다시 재지 않았다, [experiments.md](experiments.md) 9단계 끝) |
| `O-91010`, `RT-O-91009-1` | 오 다시 구일공일공, 알 티 다시 오 다시 구일공공구 다시 일 |
| `38,900원`, `10,000원` | 삼만 팔천구백 원, 만 원 (한자어 수, "일만"이 아니라 "만") |
| `9월 14일`, `6월`, `10월` | 구월 십사일, 유월, 시월 |
| `2개`, `20개`, `10시` | 두 개, 스무 개, 열 시 (세는 말 앞에서는 고유어 수) |
| `7일`, `3개월`, `5층` | 칠 일, 삼 개월, 오 층 |
| `3-5일`, `2~3개` | 삼에서 오 일, 두 개에서 세 개 (세는 말 앞의 범위는 앞 수도 고유어로 읽고 세는 말을 되풀이한다. 2026-10-13 전에는 "이에서 세 개"였다. 커밋된 음성 기록의 발화 중 이 때문에 바뀌는 것은 없다, [experiments.md](experiments.md) 5단계 끝) |
| `hajun@example.com` | 에이치 에이 제이 유 엔 골뱅이 이그잼플 닷컴 |
| `250mm`, `10%` | 이백오십 밀리미터, 십 퍼센트 |

개발용 기준선 기록에 있는 고객 발화 512개가 모두 한글, 공백, 문장부호만 남는지를 테스트가 확인한다.

## 정한 것과 이유

- **음성 모델은 옆 프로젝트에서 이미 잰 것을 쓴다.** 한국어 합성은 MeloTTS(재인식 CER 5.5%로 사람 녹음 5.4%와 비슷했다), 인식은 faster-whisper `large-v3-turbo`다. MeloTTS는 transformers 4.27을 고정하고 Windows에서 우회 세 개가 필요하다 (`speech.py`의 `_prepare_melo_on_windows`). 본체는 transformers를 쓰지 않아 한 환경에 같이 설치된다. MeloTTS의 시연 화면만 쓰는 gradio는 설치에서 뺐다
- **(2026-10-29 보탬) transformers 4.27.4 고정의 알려진 위험.** voice 그룹은 MeloTTS(git rev `2091453`)의 Requires-Dist가 `transformers==4.27.4`를 정확히 요구해 고정된다 (`pyproject.toml`의 같은 줄을 지워도 해석은 같고, 올리려면 `override-dependencies`가 필요하다. 그러면 합성 경로가 바뀐다). 이 버전에는 pip-audit 기준 고유 권고 30건이 있다 (2026-10-24. 2026-10-29에 다시 확인해 같은 30건). 함께 묶이는 패키지(transformers가 요구하는 tokenizers<0.14·huggingface-hub<1.0, MeloTTS가 요구하는 librosa==0.9.1, MeloTTS의 의존성 gruut 2.2.3이 요구하는 numpy<2.0.0. 잠금 버전은 tokenizers 0.13.3·huggingface-hub 0.36.2·librosa 0.9.1·numpy 1.26.4)에는 같은 날 권고가 없었다. 이 저장소는 고정 id `kykim/bert-kor-base`의 토크나이저·BERT만 불러 해당 코드 경로를 부르지 않고, torch 2.11의 `torch.load` 기본값(`weights_only=True`)이 pickle 경로를 막는다 (MeloTTS 체크포인트와 BERT 가중치를 읽는 두 호출 모두 `weights_only`를 주지 않아 기본값을 따른다. 환경 변수 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`를 켜면 이 막음이 풀린다). 서비스 Docker 이미지와 CI에는 설치되지 않는다 (Dockerfile은 `uv sync --frozen --no-dev`, CI는 `uv sync --frozen`이라 voice 그룹을 넣지 않는다). 이 환경에서 신뢰할 수 없는 모델·체크포인트를 열지 말 것. 왕복 캐시 키에는 transformers 버전이 없으므로, 이 핀을 바꾸면 `outputs/voice-cache/`의 예전 왕복 결과를 쓰지 않아야 한다. 버전과 모델 스냅숏은 manifest의 `voice_extra`에 남는다 (위 "음성 모델의 리비전")
- **(2026-11-01 보탬) 음성 그룹의 다른 권고.** 위 줄이 세지 않은 권고 가운데 filelock 3.32.7(13건. 대부분 잠금 파일이나 그 폴더를 같은 PC의 다른 계정이 바꿔치기·선점할 수 있을 때 성립)과 pyasn1 0.6.4(4건. BER·CER·DER 디코딩이 입력 크기의 제곱만큼 CPU를 씀)는 판을 올렸다. pyasn1은 MeloTTS → cached-path → google-cloud-storage → google-auth → pyasn1-modules로 끌려오고 이 저장소는 ASN.1을 디코딩하지 않는다 (`uv lock --upgrade-package pyasn1`로 0.6.5). filelock은 cached-path 1.8.10이 `filelock<4.0`을 요구해 고친 판(4.2.1, 하나는 4.0.9)으로 올릴 수 없어, `pyproject.toml`의 `override-dependencies`에 `filelock>=4.2.1`을 두어 그 상한을 덮었다 (잠금 4.2.6). 근거: cached-path는 `FileLock`을 상속해 생성자 인수 `read_only_ok`를 더하고 `acquire`를 감쌀 뿐이고, `FileLock`의 생성자·`acquire` 서명은 3.32.7과 4.2.6에서 같다. 2026-11-01 이 PC에서 CPU로만(모델·외부 네트워크 없이) 같은 확인 11가지를 두 판에서 돌려 결과가 같았다: cached-path 하위 클래스의 생성(기본·`read_only_ok=True`)·잠그기·풀기, 같은 경로의 둘째 잠금은 기다리지 않으면 잡지 못함, 읽기 전용 잠금 파일에서 `PermissionError`(`read_only_ok=True`면 경고와 함께 넘어감), `cached_path()`로 127.0.0.1에서 내려받기·캐시 적중·압축 풀기, huggingface_hub 0.36.2의 `WeakFileLock`(다른 잠금이 쥐고 있으면 시간 초과), torch의 `torch.utils._filelock.FileLock`, transformers·faster-whisper 불러오기. 이 저장소의 합성은 MeloTTS 기본값대로 모델을 허브에서 받으므로(`hf_hub_download`) cached-path의 잠금은 실제 합성 경로에서 불리지 않는다. filelock은 파일 잠금만 하고 합성·인식 계산에 끼지 않으며 `describe()`·`voice_extra`에도 없어, 기록된 음성 측정과의 짝짓기는 바뀌지 않는다. 남은 권고 (pip-audit 2.10.1, 2026-11-01): nltk 3.10.3 PYSEC-2026-3740 (고친 판 없음. 호출자가 정한 경로로 모델 파일을 읽고 쓰는 몇 API(`TransitionParser`, `AveragedPerceptron.save`·`load`, `PerceptronTagger.save_to_json`, `save_maxent_params`)가 NLTK의 경로 보안 설정을 우회한다. MeloTTS의 발음 변환(g2pkk·g2p_en)은 nltk 자료 폴더의 고정된 자료(cmudict, 품사 태거)만 읽고 이 저장소는 그 API에 경로를 넘기지 않는다), setuptools 81.0.0 PYSEC-2026-3447·GHSA-xmj3-9gf7-h52w (sdist를 만들 때의 `MANIFEST.in` 제외 규칙, Windows의 옛 egg 방식 wheel 설치 경로. 이 저장소는 둘 다 쓰지 않고, 고친 판 83.0.0·84.1.0은 torch 2.11.0의 `setuptools<82`가 막는다), torch 2.11.0 PYSEC-2025-194 (`torch.jit.script`의 메모리 손상. MeloTTS가 불러올 때 자기 고정 함수 두 개(`melo/commons.py`, `melo/attentions.py`)를 컴파일할 뿐 외부 코드를 넘기지 않는다. 고친 판 2.13.0은 측정 환경을 바꾸므로 올리지 않는다), 그리고 위 transformers. 서비스·CI·Docker가 설치하는 기본 그룹과 CI와 같은 조건의 가상 환경에는 권고가 없다 (CI가 push·PR마다 잠금 파일의 기본 그룹을 대조한다)
- **(2026-11-03 보탬) 남은 권고 둘은 고치지 않고 받아들인다.** nltk 3.10.3 PYSEC-2026-3740(고친 판이 없고, MeloTTS의 발음 변환 g2pkk·g2p_en은 nltk 자료 폴더의 고정된 자료만 읽으며 이 저장소는 그 API에 경로를 넘기지 않는다)과 transformers 4.27.4의 권고(MeloTTS가 정확히 고정하고, 합성 경로라 바꾸면 기록된 음성 측정과 짝이 끊긴다)는 둘 다 선택 그룹 `voice`에만 들고, 서비스·CI·Docker가 설치하는 기본 그룹에는 없어 CI의 pip-audit은 0건이다. 그래서 `--ignore-vuln`이나 판 올림 없이 그대로 둔다. 이력·수치는 바뀌지 않는다 (setuptools 81.0.0·torch 2.11.0의 권고는 위 2026-11-01 보탬대로 torch 2.11.0이 묶는다)
- **고객 → 에이전트 방향만 바꾼다.** 시뮬레이터는 자기가 쓴 글을 기억하고, 에이전트의 답은 글 그대로 시뮬레이터에 간다. 그래서 텍스트 실행과의 차이는 "에이전트가 잘못 들었다"에서만 나온다
- **같은 입력이면 같은 소리.** 합성의 표본 추출 잡음을 발화마다 정해진 seed로 고정하고, 인식은 temperature 0만 쓴다. 그래서 왕복 결과를 캐시해도 측정값이 달라지지 않는다. 캐시 키에는 말로 푼 글과 seed 말고도 모델 이름, 장치, 옵션, 패키지 버전(melotts, torch, faster-whisper, ctranslate2)이 들어간다. 버전이 바뀌면 예전 결과를 쓰지 않는다
- **음성 파일을 커밋하지 않는다.** 글과 seed로 다시 만들 수 있다
