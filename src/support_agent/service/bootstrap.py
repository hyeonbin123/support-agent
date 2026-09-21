"""Bring a service database up: connect, migrate, and fill an empty shop with the generated data."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, func, insert, select

from support_agent import db
from support_agent.seed import build_seed_engine


def _fk_on(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        path = url.removeprefix("sqlite:///")
        if path and path != url and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Requests run in worker threads; every unit of work opens its own Session.
        engine = create_engine(url, connect_args={"check_same_thread": False})
        event.listen(engine, "connect", _fk_on)
        return engine
    return create_engine(url, pool_pre_ping=True)


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("support_agent.service") / "migrations"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def migrate(engine: Engine) -> None:
    config = alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as connection:
        config.attributes["connection"] = connection  # env.py uses it instead of opening its own
        command.upgrade(config, "head")


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


def prepare_database(url: str, *, load_seed: bool = True) -> Engine:
    engine = make_engine(url)
    migrate(engine)
    if load_seed:
        copy_seed(engine)
    return engine
