"""Bring a service database up: connect, migrate, and fill an empty shop with the generated data."""

# [파일 역할] 서비스 DB를 쓸 수 있는 상태로 올린다: 엔진 생성 → Alembic 마이그레이션 → 쇼핑몰이 비어 있으면
# 평가와 같은 생성 데이터 복사.
# [왜 필요한가] 평가는 에피소드마다 메모리 SQLite 복사본을 쓰지만, 서비스는 데이터가 남아야 하므로
# PostgreSQL(Compose)이나 SQLite 파일을 쓴다. 어느 쪽이든 같은 테이블 정의와 같은 생성 데이터에서 시작해야
# 평가에서 잰 도구 동작이 서비스에서도 그대로 나온다(README "구조").
# [연결] service/app.py의 create_app과 mcp_server.py가 prepare_database를 부르고, 테스트(test_service.py,
# test_mcp_server.py, test_postgres.py)도 같은 함수로 DB를 만든다.

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# alembic: 스키마를 코드(migrations/versions/)로 관리하고 upgrade head로 올린다. 이미 올라간 DB에 다시 돌려도
# 남은 버전만 적용되므로 앱이 뜰 때마다 불러도 안전하다.
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, func, insert, select

# db: 쇼핑몰 테이블 선언(복사할 테이블 목록과 순서). seed.build_seed_engine: 평가와 같은 seed 고정 생성
# 데이터를 담은 메모리 DB를 만든다.
from support_agent import db
from support_agent.seed import build_seed_engine


# SQLite는 연결마다 외래 키 검사가 꺼진 채 시작하므로, 새 연결이 열릴 때마다 foreign_keys 설정을 켠다.
# PostgreSQL과 같은 무결성 검사를 받게 하려는 것이다.
def _fk_on(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# DB 주소에 맞는 SQLAlchemy 엔진을 만든다.
# - SQLite 파일이면 상위 폴더(기본 outputs/)를 먼저 만든다. 메모리 DB나 경로가 없는 주소는 건너뛴다
# - PostgreSQL 등은 pool_pre_ping으로 풀에서 꺼낸 연결이 살아 있는지 쓰기 전에 확인하고, 끊겼으면 새로
#   연결한다
def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        path = url.removeprefix("sqlite:///")
        if path and path != url and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Requests run in worker threads; every unit of work opens its own Session.
        # check_same_thread=False: SQLite 연결을 만든 스레드 밖에서도 쓰게 한다. FastAPI는 동기 처리기를 작업
        # 스레드에서 돌리기 때문이다. 대신 작업 단위마다 Session을 따로 열어 한 연결을 동시에 나눠 쓰지
        # 않는다.
        engine = create_engine(url, connect_args={"check_same_thread": False})
        event.listen(engine, "connect", _fk_on)
        return engine
    return create_engine(url, pool_pre_ping=True)


# alembic.ini 파일 없이 코드로 Alembic 설정을 만든다. 마이그레이션 스크립트는 패키지
# 안(support_agent.service/migrations)에 있으므로 importlib.resources로 경로를 찾아, 설치된 패키지(Docker
# 이미지)에서도 같은 위치를 쓴다.
# Alembic 설정은 configparser라서 '%'가 보간 문자로 읽힌다. URL 인코딩된 비밀번호의 '%'가 깨지지 않게 '%%'로
# 바꿔 넣는다.
def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("support_agent.service") / "migrations"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


# 스키마를 최신(head)으로 올린다. 이미 연 연결을 config.attributes로 env.py에 넘겨, env.py가 새 엔진을 만들지
# 않고 같은 연결·같은 트랜잭션에서 마이그레이션한다.
# render_as_string(hide_password=False): 엔진 URL을 그냥 문자열로 바꾸면 비밀번호가 ***로 가려지므로 실제 값을
# 넣는다.
def migrate(engine: Engine) -> None:
    config = alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as connection:
        config.attributes["connection"] = connection  # env.py uses it instead of opening its own
        command.upgrade(config, "head")


# 고객 행이 하나라도 있으면 이미 데이터가 있는 DB로 보고 건드리지 않는다. 그래서 서비스를 다시 띄워도 그동안의
# 주문 변경이 지워지지 않는다.
# sorted_tables는 외래 키 의존 순서(부모 → 자식)라, 자식 행이 아직 들어가지 않은 부모 행을 가리키는 일 없이
# 넣을 수 있다. 한 트랜잭션(engine.begin)이므로 중간에 실패하면 반쯤 채운 상태가 남지 않는다.
def copy_seed(engine: Engine) -> bool:
    """Copy the generated shop data into an empty shop. Returns False when there were customers already."""
    with engine.begin() as target:
        if target.execute(select(func.count()).select_from(db.Customer.__table__)).scalar_one():
            return False
        with build_seed_engine().connect() as source:
            for table in db.Base.metadata.sorted_tables:  # parents before children
                rows = [dict(row) for row in source.execute(select(table)).mappings()]
                if rows:
                    target.execute(insert(table), rows)
    return True


# 서비스·MCP 서버가 쓰는 진입점. load_seed=False면 빈 쇼핑몰로 둔다(설정 SUPPORT_AGENT_LOAD_SEED).
def prepare_database(url: str, *, load_seed: bool = True) -> Engine:
    engine = make_engine(url)
    migrate(engine)
    if load_seed:
        copy_seed(engine)
    return engine
