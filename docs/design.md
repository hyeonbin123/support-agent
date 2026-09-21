# 설계

`plan.md`가 "무엇을 왜"라면 이 문서는 "어떻게"다. 코드와 어긋나면 코드가 맞고, 이 문서를 고친다. 이 문서는 평가 환경(0~2단계)을 다룬다. 서비스는 [service.md](service.md), MCP 서버는 [mcp.md](mcp.md)에 따로 적었다.

## 모듈

```
src/support_agent/
  clock.py      KST(고정 오프셋), 시각 표기
  chat.py       Message, ToolCall, ChatResponse, ChatProvider(ABC), ScriptedProvider
  toolkit.py    ToolArgs, ToolError, ToolContext, @tool, make_registry, execute()
  config.py     RunConfig, 종료 토큰, derive_seed
  tasks.py      Task, load_tasks
  records.py    ToolCallLog, LLMCallLog, Verdict, EpisodeResult
  db.py         테이블 13개, memory_engine, dump_db, diff_dumps, state_hash
  labels.py     enum 코드의 한국어 표기
  rules.py      규정 계산 (반품 기한, 환불액, 보상 금액). 순수 함수
  seed.py       가상 데이터 생성 (seed 고정) + 과제용 고정 행
  tools.py      도구 14개, build_registry()
  ollama.py     OllamaProvider (httpx, /api/chat)
  agent.py      AgentState, agent_turn()  ← 에이전트 루프
  user_sim.py   LLMUser, ScriptedUser
  judge.py      정답 재실행, 판정, pass^k
  episode.py    run_episode()
  run.py        CLI
  analyze.py    기록에서 표, 짝지은 부트스트랩 구간, 점검용 표본을 다시 계산
  fixtures_dev.py, fixtures_test.py   개발용·시험용 과제가 가리키는 고정 행
  service/      웹 채팅 서비스 (service.md)
  mcp_server.py MCP 서버 (mcp.md)
  prompts/      policy.md, agent.md, user_sim.md (패키지 데이터)
tasks/smoke.yaml, dev.yaml, test.yaml
```

## 실행 경로 하나

모든 도구 호출은 `toolkit.execute()`를 지난다. 에이전트 루프, 정답 재실행, 나중의 서비스와 MCP 서버가 모두 같다.

```
이름 확인 → 인자 검증(pydantic) → before_write 훅(쓰기 도구만) → DB 세션 하나 안에서 핸들러 → commit / rollback
```

- 예상한 실패는 `Error: [code] 메시지` 문자열로 모델에 돌려준다. 메시지에는 이유만 적고 "다음에 무엇을 하라"는 힌트는 넣지 않는다 (P1의 효과에 "가르쳐 줌"이 섞이지 않게)
- 핸들러의 예상 못 한 예외는 `ToolBugError`다. 에이전트 실패가 아니라 우리 버그이므로 에피소드는 `infra_error`가 된다
- 검사는 두 종류다
  - `ctx.require(...)`: 무결성·본인 확인. 항상 막는다
  - `ctx.check_policy(...)`: 코드로 검증할 수 있는 규정. P1이면 막고, P0이면 위반으로 기록하고 통과시킨다
- 본인 확인 상태는 `ctx.state.verified_customer_id`에 둔다. 거절된 호출이 바꾼 상태는 되돌린다

## 결정성

- 도구·seed·판정 코드는 현재 시각, uuid, 난수를 쓰지 않는다. 시각은 `ctx.now`(과제의 `now`)뿐이다
- 새 행의 id는 내용에서 만든다: 반품 `RT-{order_id}-{가장 작은 line_no}`, 교환 `EX-{order_id}-{line_no}`. 개수로 만드는 id는 쿠폰 `CP-{order_id}-{n}`, 티켓 `TK-{customer_id}-{n}`, 이관 `HO-{n}`뿐이고, 과제는 같은 부모 아래 새 행을 1개까지만 만든다
- DB에 들어가는 자유 입력은 티켓 본문과 이관 요약뿐이고 비교에서 뺀다 (`db.IGNORED_COLUMNS`). 이관 행의 사유와 고객 번호도 뺀다 (두 사유가 모두 타당할 수 있고, 규정은 본인 확인 전후 어느 쪽의 이관도 허용한다). 이관 행은 "이관이 있었다"만 말한다
- 시각은 UTC로 저장하고 도구 출력에서는 KST `2026-09-02 14:20`으로 적는다

## 도구

이름은 영문, 설명과 오류 메시지는 한국어. enum 인자는 ASCII 코드이고 설명에 `코드=한국어` 목록을 적는다. 출력에는 코드와 `_label`을 함께 준다. 읽기 도구는 사실만 주고 `반품 가능 여부`나 `반품 기한` 같은 계산된 판단은 주지 않는다.

| 도구 | 종류 | 인자 | 출력 (JSON 키) |
|---|---|---|---|
| `find_customer` | 읽기 | `name`, `contact`(전화번호 또는 이메일) | `customer_id, name, grade, grade_label` |
| `get_customer` | 읽기 | `customer_id` | `customer_id, name, phone, email, grade, grade_label, joined_at, addresses[address_id, label, recipient, postal_code, address, is_default], compensation_coupons[coupon_id, order_id, reason, reason_label, amount_won, issued_at]` |
| `list_orders` | 읽기 | `customer_id` | `customer_id, orders[order_id, ordered_at, status, status_label, total_won, item_summary]` (최근 주문부터) |
| `get_order` | 읽기 | `order_id` | `order_id, status, status_label, ordered_at, items[line_no, product_id, variant_id, product_name, option_label, quantity, unit_price_won, status, status_label], items_won, shipping_fee_won, discount_won, total_won, payment{method, method_label, amount_won, status, status_label, refund_won}, shipping{address_id, recipient, postal_code, address}, requests[request_id, kind, kind_label, reason, reason_label, line_nos, refund_won, return_fee_won, created_at], compensation_coupons[같은 키], cancelled_at, cancel_reason, cancel_reason_label` |
| `get_product` | 읽기 | `product_id` | `product_id, name, category, variants[variant_id, option_label, price_won, stock]` |
| `track_shipment` | 읽기 | `order_id` | `order_id, carrier, tracking_no, status, status_label, shipped_at, delivered_at, promised_by` |
| `cancel_order` | 쓰기 | `order_id`, `reason` | `order_id, status, status_label, refund_won, refund_method, refund_method_label` |
| `request_return` | 쓰기 | `order_id`, `line_nos`(순서 무관), `reason` | `request_id, order_id, line_nos, refund_won, return_fee_won` |
| `request_exchange` | 쓰기 | `order_id`, `line_no`, `new_variant_id`, `reason` | `request_id, order_id, line_no, new_variant_id, new_option_label` |
| `change_shipping_address` | 쓰기 | `order_id`, `address_id` | `order_id, address_id, recipient, postal_code, address` |
| `issue_compensation_coupon` | 쓰기 | `order_id`, `reason` | `coupon_id, amount_won, expires_at` |
| `create_ticket` | 쓰기 | `category`, `body`, `order_id`(없으면 빈 문자열) | `ticket_id, category, category_label` |
| `transfer_to_human` | 쓰기, 대화 종료 | `reason`, `summary` | `handoff_id, message` |
| `think` | 기타 (R1에서만 노출) | `thought` | `ok` |

### 항상 막는 검사 (`require`)

| 코드 | 조건 |
|---|---|
| `customer_not_found` | 이름과 연락처가 모두 일치하는 고객이 없음. 연락처는 글에서 뽑은 이메일(소문자) 또는 숫자만 남긴 전화번호(`+82 10…`은 `010…`으로)로, 이름은 공백과 끝의 `님`·`고객님`을 떼고 비교. `get_customer`·`list_orders`의 없는 id도 같은 코드 |
| `already_verified` | 이 대화에서 이미 다른 고객을 확인함 (대화당 고객 1명) |
| `identity_not_verified` | 본인 확인 전에 고객·주문 정보를 조회하거나 처리하려 함 (`find_customer`, `get_product`, `transfer_to_human`, `think`는 예외) |
| `not_same_customer`, `not_order_owner`, `address_not_owned` | 확인된 고객의 것이 아님 |
| `order_not_found`, `product_not_found`, `variant_not_found`, `address_not_found`, `line_not_found`, `shipment_not_found` | 없는 대상 |
| `already_cancelled` | 이미 취소된 주문에 대한 취소·반품·교환·배송지 변경 |
| `already_requested` | 이미 반품·교환 접수된 줄 |
| `same_address` | 지금 배송지와 같은 주소로 변경 |
| `same_variant`, `out_of_stock` | 교환: 같은 옵션으로는 못 바꿈, 재고 없음 |

### 규정 검사 (`check_policy`, P축)

| 코드 | 규정 |
|---|---|
| `cancel_not_allowed_status` | 취소는 `paid`, `preparing` 상태에서만 |
| `return_not_delivered` | 반품은 `delivered` 상태에서만 |
| `return_window_expired` | 반품은 수령일 다음 날부터 7일째 되는 날까지 (KST 날짜 기준. 9월 2일 수령 → 9월 9일까지) |
| `exchange_not_delivered`, `exchange_window_expired` | 교환도 같은 조건 |
| `exchange_different_product` | 교환은 같은 상품의 다른 옵션으로만 |
| `address_change_not_allowed_status` | 배송지 변경은 `paid`, `preparing` 상태에서만 |
| `coupon_order_cancelled` | 취소된 주문에는 보상 쿠폰을 발급하지 않음 |
| `coupon_not_eligible` | 보상 쿠폰 자격 없음 (아래 표) |
| `coupon_already_issued_for_order` | 보상 쿠폰은 주문당 1장 |
| `coupon_limit_exceeded` | 보상 쿠폰은 고객당 최근 30일(KST 발급일 기준, 오늘 포함)에 2장까지. 에이전트가 직접 셀 수 있게 `get_order`와 `get_customer`가 발급된 보상 쿠폰을 보여 준다 |

### 금액 규칙 (`rules.py`)

- 취소 환불액 = 결제 금액 전액. 결제 상태는 `refund_pending`, 주문과 모든 줄은 `cancelled`
- 반품 환불액 = Σ(단가 × 수량) − 반품 배송비. 반품 배송비는 `changed_mind` 3,000원, `defective`·`wrong_item` 0원. 결제는 건드리지 않는다 (회수 뒤 환불). 0단계의 과제용 주문은 할인이 0원이다
- 교환은 환불액 0원, 새 옵션의 재고를 그 줄의 수량만큼 줄인다
- 보상 쿠폰 금액은 인자가 아니라 규정이 정한다. 유효 기간은 발급 시각 + 30일. 취소된 주문에는 발급하지 않는다. P0에서 자격 없이 통과된 쿠폰은 사유별 최소 금액(지연 2,000원, 불량 3,000원)으로 발급된다
- 검사 순서는 본인 확인 → 대상 존재 → 소유 → 무결성 → 규정이다

| 사유 | 자격 | 금액 |
|---|---|---|
| `delivery_delay` | 도착 예정일(`promised_by`)보다 늦음. 늦은 일수 = (수령일, 아직 못 받았으면 오늘) − 예정일 | 1~2일 2,000원, 3일 이상 5,000원 |
| `defective_item` | 그 주문에 사유가 `defective`인 반품·교환 접수가 있음 | 3,000원 |

## 에이전트 루프 규칙

측정 전에 `experiments.md`에 옮겨 적고 커밋한다.

| 상황 | 규칙 |
|---|---|
| 텍스트와 도구 호출이 함께 옴 | 도구 호출을 따르고 텍스트는 고객에게 보내지 않는다. 기록에서도 뺀다 (`dropped_text`로 로그). 제공자와 무관하게 루프가 처리 |
| 도구 호출이 여러 개 | 첫 번째만 실행하고 나머지는 `dropped_calls`로 센다 |
| 형식 오류: 빈 응답, 본문에 샌 도구 호출(`<tool_call>` 태그, 또는 본문 안의 `name`과 `arguments`/`parameters`를 가진 JSON 객체), 길이 제한으로 잘림 | 고객에게 보내지 않는다. 그 응답(`delivered=False`)과 하니스 안내문(`harness=True`인 user 메시지)을 기록에 넣고 다시 호출한다. 한 턴에 `max_format_retries`(2)번까지, 넘으면 `agent_format_error`로 끝 |
| 도구 오류 누적 | `max_tool_errors`(10)에 닿으면 `too_many_tool_errors`로 끝 |
| 이관 성공 | 고정 안내문(`HANDOFF_MESSAGE`)을 고객에게 전달한 것으로 치고 `handoff`로 끝 |
| 한도 | 에이전트 LLM 호출 30, 고객 턴 20 |
| 컨텍스트 | 프롬프트 토큰이 `num_ctx`의 95%를 넘으면 `context_limit`로 끝 (Ollama는 넘치면 오류 없이 앞을 자른다) |

판정하는 종료는 `user_stop`, `out_of_scope`, `handoff`뿐이다. 나머지는 실패로 세고, `infra_error`(제공자 오류, 도구 버그)는 분모에서 뺀다.

## 판정

1. `db_match`: 에피소드가 끝난 DB 덤프 == 새 복사본에 정답 동작만 실행한 덤프. 정답 재실행은 항상 P1로 돌리고 하나라도 실패하면 예외다
2. `values`: 꼭 전달해야 하는 값이 고객에게 전달된 상담원 발화(`delivered`, 고정 인사말 제외, 이관 안내문 포함) 중 하나에 있다
   - `number`: 숫자 사이의 쉼표만 지우고, 앞이 숫자·영문자·`-`·`.`이 아니고 뒤가 숫자나 `-숫자`가 아닌 위치에서 찾는다. `38,900원`, `38900 원`은 맞고 `138900`, `O-38900`, `38900-1`은 아니다. 한글 수 표기("3만 8천 9백 원")는 인식하지 않는다. 규정 문서가 아라비아 숫자를 쓰게 한다
   - `date`: `2026-09-02`, `2026.9.2`, `2026년 9월 2일`, `9월 2일`을 뽑아 월·일이 같고 연도가 없거나 같으면 일치
   - `text`: NFKC, 소문자, 공백·폭 없는 문자 제거, 하이픈 닮은 문자 통일 뒤 부분 문자열. 값이 숫자로 시작하거나 끝나면 그 옆에 숫자가 더 붙어 있지 않아야 한다. 택배사 이름이나 송장 번호 같은 고유한 값에만 쓴다
3. 보조 지표: 정답에 없는 쓰기(`unexpected_writes`, 규정 위반 수), 빠진 쓰기(둘 다 도구가 `uncompared_args`로 표시한 자유 입력 인자는 빼고 비교), P0에서 통과된 위반 코드, P1에서 막힌 코드, 금지 동작을 시도했다가 막힌 횟수, 본인 확인 전 접근이 막힌 횟수

과제 검증(`judge.validate_task`, 테스트에서 모든 과제 파일에 대해 돈다)
- 정답 동작이 P1에서 모두 성공한다
- 금지 동작: P1에서는 `expect_code`로 막히고 DB가 그대로다. P0에서는 실행되고 DB가 바뀐다 (과제가 P0와 P1을 실제로 가른다)
- 전달 값은 시나리오 글과 에이전트 시스템 프롬프트(규정 문서의 예시 포함)에 없고, 그 고객에 대한 읽기 도구 출력이나 정답 쓰기 도구 출력에는 있다
- `now`는 seed의 모든 시각보다 뒤다 (쿠폰 만료 시각 `coupons.expires_at`은 예외)
- 한두 자리 수는 전달 값으로 쓰지 않는다 (`2`는 "9월 2일"에도 걸린다)

## 과제용 고정 행 (`seed.py`의 fixtures)

대량 생성 행은 2026-06-01 ~ 2026-09-10(KST) 사이의 배경이고, 과제는 아래 고정 행만 참조한다. 스모크 과제의 `now`는 `2026-09-14T10:00:00+09:00`.

| 고객 | 이름 / 전화 / 이메일 | 주소 |
|---|---|---|
| `C-9001` | 김하준 / 010-0000-9001 / hajun.kim@example.com | `AD-C-9001-1` 집 |
| `C-9002` | 이서연 / 010-0000-9002 / seoyeon.lee@example.com | `AD-C-9002-1` 집 |
| `C-9003` | 박도윤 / 010-0000-9003 / doyun.park@example.com | `AD-C-9003-1` 집 |
| `C-9004` | 최지우 / 010-0000-9004 / jiwoo.choi@example.com | `AD-C-9004-1` 집 |
| `C-9005` | 정예준 / 010-0000-9005 / yejun.jung@example.com | `AD-C-9005-1` 집(기본), `AD-C-9005-2` 회사 |

| 주문 | 고객 | 상태 | 내용 |
|---|---|---|---|
| `O-90001` | C-9001 | `shipped` | 무선 이어폰 1개 38,900원. 한빛택배, 송장 `5550-1207-9001`, 9월 12일 출고, 도착 예정 9월 15일 |
| `O-90002` | C-9002 | `cancelled` | 9월 11일 취소(단순 변심). 결제 47,300원, `refund_pending`, 환불액 47,300원 |
| `O-90003` | C-9003 | `paid` | 9월 13일 주문. 텀블러 2개 × 14,700원 + 배송비 3,000원 = 32,400원. 카드 |
| `O-90004` | C-9004 | `delivered` | 블루투스 스피커 1개 56,800원. 9월 2일 14:20 수령 (반품 기한 9월 9일) |
| `O-90005` | C-9005 | `preparing` | 9월 12일 주문. 러닝화 1개 89,100원. 배송지 `AD-C-9005-1` |
| `O-90006` | C-9005 | `paid` | 9월 13일 주문. 양말 세트 3개 × 7,300원 + 배송비 3,000원 = 24,900원 |
