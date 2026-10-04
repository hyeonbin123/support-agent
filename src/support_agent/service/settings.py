"""Settings of the service, read from environment variables (prefix SUPPORT_AGENT_)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, model_validator

PREFIX = "SUPPORT_AGENT_"
# The generated shop data is written around this moment (the tasks use it too). With the wall clock every
# order would soon be past its return window, so the demo runs on a fixed clock unless SUPPORT_AGENT_NOW=real.
DEMO_NOW = "2026-09-14T10:00:00+09:00"
# Since 2026-10-04 (docs/experiments.md, stage 8): test pass^1 55.6% against 18.1% for qwen2.5 7B, measured
# with thinking off. It also made 13 writes the tasks did not ask for against 6, so the approval gate stays.
# MODEL=qwen2.5:7b-instruct brings the earlier model back.
DEFAULT_MODEL = "qwen3.5:4b"
_THINK = {"on": True, "off": False, "omit": None}


class Settings(BaseModel):
    database_url: str = "sqlite:///outputs/service.db"
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    # Ollama's `think` field on every agent request. Left unset it is "off" for the default model (measured
    # so) and "omit" (not sent) for any other: models without a thinking mode refuse the field.
    think: Literal["on", "off", "omit"] = "off"
    num_ctx: int = 16384
    # The service refuses what the rules forbid (P1). Whether that raises the success rate is a question of
    # docs/experiments.md; here a wrong refund costs money, so the tools check whatever code can check.
    policy: Literal["P0", "P1"] = "P1"
    # C1: a reply that says work is done which no tool result of the conversation shows is never delivered.
    # Off by default: on the development tasks C1 cost −2.1%p and one more unexpected write, C2 −5.2%p, so
    # neither met the rule written before measuring (docs/experiments.md, stage 7). CLAIMS=C1 turns it on.
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

    @model_validator(mode="before")
    @classmethod
    def _think_follows_the_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("think") in (None, ""):
            model = data.get("model") or DEFAULT_MODEL
            data = {**data, "think": "off" if model == DEFAULT_MODEL else "omit"}
        return data

    def agent_think(self) -> bool | None:
        """The value OllamaProvider sends as `think`; None sends nothing."""
        return _THINK[self.think]

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
