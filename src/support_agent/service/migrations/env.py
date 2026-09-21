"""Alembic environment: the shop tables and the service tables are migrated together."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from support_agent import db
from support_agent.service import store

target_metadata = [db.Base.metadata, store.ServiceBase.metadata]


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=False)
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:  # called from bootstrap.migrate with an open connection
    _run(connection)
else:  # called from the alembic command line
    engine = create_engine(context.config.get_main_option("sqlalchemy.url"))
    with engine.connect() as own:
        _run(own)
        own.commit()
