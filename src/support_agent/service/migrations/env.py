"""Alembic 환경: 쇼핑몰 테이블과 서비스 테이블을 함께 마이그레이션한다."""

# [파일 역할] Alembic이 마이그레이션을 돌릴 때 실행하는 환경 스크립트다. 어떤 연결로, 어떤 테이블 선언을
# 기준으로 돌릴지 정한다.
# [설계 이유] 쇼핑몰 테이블(db.Base)과 서비스 테이블(store.ServiceBase)은 메타데이터를 나눴지만 같은 DB에
# 있으므로 마이그레이션 하나로 함께 관리한다.
# [연결] 보통은 service/bootstrap.py의 migrate가 연 연결을 넘겨받아 돈다. alembic 명령줄로 부를 때만 직접
# 엔진을 만든다.

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from support_agent import db
from support_agent.service import store

# 비교 기준이 되는 두 메타데이터. tests/test_postgres.py가 같은 목록으로 "마이그레이션 결과 = 선언한
# 테이블"인지 확인한다(docs/service.md).
target_metadata = [db.Base.metadata, store.ServiceBase.metadata]


# 주어진 연결에서 마이그레이션을 실행한다. render_as_batch=False: SQLite에서 표를 통째로 다시 만드는 batch
# 모드를 쓰지 않는다(지금 마이그레이션은 표를 새로 만들기만 한다).
def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=False)
    with context.begin_transaction():
        context.run_migrations()


# bootstrap.migrate가 넘긴 연결이 있으면 그것을 쓴다. 그 연결의 트랜잭션은 bootstrap 쪽 engine.begin()이
# 커밋하므로 여기서는 커밋하지 않는다.
# 명령줄에서 부른 경우에는 설정의 주소로 엔진을 만들고 끝에 직접 커밋한다.
connection = context.config.attributes.get("connection")
if connection is not None:  # bootstrap.migrate가 열린 연결과 함께 부른 경우
    _run(connection)
else:  # alembic 명령줄에서 부른 경우
    engine = create_engine(context.config.get_main_option("sqlalchemy.url"))
    with engine.connect() as own:
        _run(own)
        own.commit()
