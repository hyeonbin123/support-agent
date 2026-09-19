"""Task files: what the simulated customer wants and what a correct ending looks like."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from support_agent.paths import TASKS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAction(_Strict):
    tool: str
    args: dict[str, Any]


class ForbiddenAction(ToolAction):
    expect_code: str  # the policy code that has to refuse this call under P1


class RequiredValue(_Strict):
    """Something the agent has to tell the customer. It must come from a tool result, never from the scenario
    or the policy text, so that an agent that did not look anything up cannot pass."""

    kind: Literal["number", "date", "text"]
    value: int | date | str
    label: str

    @model_validator(mode="after")
    def _kind_matches_value(self) -> RequiredValue:
        expected = {"number": int, "date": date, "text": str}[self.kind]
        if type(self.value) is not expected:
            raise ValueError(f"{self.label}: kind {self.kind} needs a {expected.__name__} value")
        return self


class UserScenario(_Strict):
    persona: str = ""
    reason: str  # why the customer is calling
    known: str  # facts the customer may reveal, including every choice that decides a write argument
    unknown: str = ""  # facts the customer must say they do not know
    rules: str = ""  # how to react (what to ask, when to accept a refusal, when to stop)

    def all_text(self) -> str:
        return "\n".join([self.persona, self.reason, self.known, self.unknown, self.rules])


class Task(_Strict):
    id: str
    type: Literal["lookup", "action", "refusal", "composite"]
    purpose: str
    now: AwareDatetime  # fixed clock of the episode
    customer_id: str  # never shown to the simulator; used by the gold replay
    user: UserScenario
    gold_actions: list[ToolAction] = []  # write calls only, in one valid order
    gold_verified: bool = True  # False: the gold replay runs without a verified customer
    forbidden_actions: list[ForbiddenAction] = []
    required_values: list[RequiredValue] = []
    notes: str = ""

    @model_validator(mode="after")
    def _shape(self) -> Task:
        if self.type == "lookup" and self.gold_actions:
            raise ValueError(f"{self.id}: a lookup task changes nothing, so it has no gold actions")
        if self.type in ("lookup", "refusal") and not self.required_values:
            raise ValueError(
                f"{self.id}: {self.type} tasks need a required value, or doing nothing would pass"
            )
        if self.type == "refusal" and not self.forbidden_actions:
            raise ValueError(f"{self.id}: a refusal task names the call that must be refused")
        if self.type in ("action", "composite") and not self.gold_actions:
            raise ValueError(f"{self.id}: {self.type} tasks need gold actions")
        return self

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def load_tasks(name_or_path: str | Path) -> list[Task]:
    """Load `tasks/<name>.yaml` (or a path). utf-8-sig tolerates a BOM left by Windows editors."""
    path = Path(name_or_path)
    if not path.suffix:
        path = TASKS / f"{name_or_path}.yaml"
    tasks = [Task.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8-sig"))]
    ids = [task.id for task in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate task ids: {duplicates}")
    return tasks
