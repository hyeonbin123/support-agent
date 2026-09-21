# MCP 서버 (4단계)

에이전트가 쓰는 도구 등록부를 그대로 MCP 서버로 내놓는다. 도구 이름, 설명, 인자 스키마는 에이전트에게 보여 주는 것과 같은 정의에서 나온다 (`ToolSpec.schema()`). MCP 클라이언트(Claude Desktop, Claude Code 등)에서 붙으면 그 클라이언트의 모델이 상담원 역할을 한다.

## 실행

```bash
uv run python -m support_agent.mcp_server                      # stdio, 읽기 도구만
uv run python -m support_agent.mcp_server --scope write        # DB를 바꾸는 도구까지
uv run python -m support_agent.mcp_server --http --port 8063   # Streamable HTTP
```

- DB는 서비스와 같은 설정을 읽는다 (`SUPPORT_AGENT_DATABASE_URL`, 기본은 `outputs/service.db`). Compose로 띄운 PostgreSQL을 가리키면 채팅 서비스와 같은 DB, 같은 승인 대기열, 같은 감사 로그를 쓴다: `postgresql+psycopg://support:<비밀번호>@127.0.0.1:55462/support`
- HTTP는 `SUPPORT_AGENT_MCP_TOKEN`이 없으면 뜨지 않는다. 모든 요청에 `Authorization: Bearer <토큰>`이 있어야 한다. 기본으로 127.0.0.1에만 연다
- 클라이언트 설정 예 (stdio):

```json
{
  "mcpServers": {
    "support-agent": {
      "command": "uv",
      "args": ["run", "--directory", "<저장소 경로>", "python", "-m", "support_agent.mcp_server", "--scope", "write"]
    }
  }
}
```

## 정한 것과 이유

- **읽기와 쓰기를 나눈다.** 기본은 읽기 도구 6개만 보인다. `--scope write`여야 취소·반품·교환·배송지 변경·쿠폰·티켓·이관이 보이고, 읽기 서버에 쓰기 도구를 호출하면 `not_allowed`로 거절한다. 도구마다 MCP 주석(`readOnlyHint`, `destructiveHint`)을 단다. `think` 도구는 평가용 장치라 내놓지 않는다.
- **서비스와 같은 길을 지난다.** 호출은 `ChatService.call_tool`로 들어가 `toolkit.execute()`를 지난다. 그래서 도구가 규정을 막고(P1), 환불액이 기준 이상이면 승인 대기열로 가고(`approval_required`, 관리 화면에서 승인하면 그때 실행), 호출마다 감사 로그가 남는다. MCP로 붙었다고 사람 승인을 건너뛸 수 없다.
- **본인 확인 상태는 서버가 들고 있다.** `find_customer`로 확인한 고객은 `new_conversation`을 부를 때까지 유지된다. 상담 하나는 고객 한 명의 것이라(채팅과 같음) 다른 고객으로 바꾸려면 `new_conversation`으로 새 상담을 시작한다. MCP 연결에 상태를 묶지 않은 이유: 지금 프로토콜 개정판은 상태가 없는 방식이고, SDK 2.x는 요청마다 서버 세션 객체를 새로 만든다. stdio에서는 클라이언트가 서버 프로세스를 직접 띄우므로 "프로세스 하나 = 클라이언트 하나"다. HTTP 서버는 토큰 하나를 쓰는 담당자 한 명을 가정한다.
- **크기 제한.** 한 호출의 인자는 JSON으로 20,000바이트까지, HTTP 요청 본문은 256 KB까지다. 도구의 자유 서술 인자(티켓 본문, 이관 요약)에는 길이 제한이 없어서 MCP 경계에서 막는다. 도구 스키마를 고치면 에이전트가 보는 도구 정의가 바뀌어 지금까지의 측정과 비교할 수 없게 된다
- **오류는 결과로 돌려준다.** 도구가 거절하면 `isError: true`와 에이전트가 받는 것과 같은 문장(`Error: [코드] 설명`)을 돌려준다. 클라이언트의 모델이 읽고 고칠 수 있게 하기 위해서다.
- 저수준 `Server`를 쓴다. 도구가 등록부에서 동적으로 만들어지므로 함수마다 데코레이터를 다는 고수준 API보다 맞다. SDK는 `mcp>=2.2,<3`으로 고정했다.

## 확인한 것

- SDK의 메모리 내 클라이언트로 하는 테스트 6개 (`tests/test_mcp_server.py`): 범위별 도구 목록과 스키마, 본인 확인 전 거절, 다른 고객 주문 거절, 읽기 서버의 쓰기 거절, 쓰기 실행과 감사 로그, 큰 환불의 승인 대기와 승인 후 실행, HTTP 토큰
- 실제 stdio: 서버를 하위 프로세스로 띄우고 SDK 클라이언트로 목록 조회, 본인 확인, 주문 조회, 주문 취소까지 확인했다
