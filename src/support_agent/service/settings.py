"""Settings of the service, read from environment variables (prefix SUPPORT_AGENT_)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field

PREFIX = "SUPPORT_AGENT_"
# The generated shop data is written around this moment (the tasks use it too). With the wall clock every
# order would soon be past its return window, so the demo runs on a fixed clock unless SUPPORT_AGENT_NOW=real.
DEMO_NOW = "2026-09-14T10:00:00+09:00"


class Settings(BaseModel):
    database_url: str = "sqlite:///outputs/service.db"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b-instruct"
    num_ctx: int = 16384
    # The service refuses what the rules forbid (P1). Whether that raises the success rate is a question of
    # docs/experiments.md; here a wrong refund costs money, so the tools check whatever code can check.
    policy: Literal["P0", "P1"] = "P1"
    # C1: a reply that says work is done which no tool result of the conversation shows is never delivered
    # (docs/experiments.md, stage 7, also says why this default is what it is).
    claims: Literal["C0", "C1", "C2"] = "C0"
    now: AwareDatetime | Literal["real"] = Field(default_factory=lambda: datetime.fromisoformat(DEMO_NOW))
    approval_refund_won: int = 100_000  # refunds from this amount wait for a person
    admin_token: str = ""  # "" disables the admin API
    max_message_chars: int = Field(default=1000, ge=1, le=4000)  # the request model stops at 4000
    max_agent_calls_per_turn: int = 12
    max_turns_per_session: int = 40
    load_seed: bool = True  # fill an empty shop database with the generated data
    expose_openapi: bool = False  # serve /openapi.json (for API scanners and clients; off by default)
    voice: bool = False  # speech in and out (needs the `voice` dependency group)
    tts_device: Literal["cuda", "cpu"] = "cuda"
    stt_device: Literal["cuda", "cpu"] = "cuda"
    max_audio_bytes: int = 5_000_000  # about five minutes of browser-recorded speech
    max_tts_chars: int = Field(default=600, ge=1, le=4000)
    max_audio_seconds: int = 60  # a small file can hold hours of low-bitrate speech
    max_concurrent_turns: int = 4  # turns (and recognitions) running at once, over all sessions

    def clock(self) -> datetime:
        return datetime.now(UTC) if self.now == "real" else self.now

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ) if env is None else env
        values = {
            name: env[PREFIX + name.upper()]
            for name in Settings.model_fields
            if env.get(PREFIX + name.upper(), "") != ""
        }
        return Settings.model_validate(values)
