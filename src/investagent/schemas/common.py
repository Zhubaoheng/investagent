"""Shared enums, base models, and evidence types used across all agent schemas."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class EvidenceType(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"


class EvidenceItem(BaseModel, frozen=True):
    content: str
    source: str
    evidence_type: EvidenceType


class AgentMeta(BaseModel, frozen=True):
    model_config = ConfigDict(protected_namespaces=())

    agent_name: str
    timestamp: datetime
    model_used: str
    token_usage: int


class StopSignal(BaseModel, frozen=True):
    should_stop: bool
    reason: str


class BaseAgentOutput(BaseModel, frozen=True):
    meta: AgentMeta
    stop_signal: StopSignal | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_json_string_lists(cls, data: Any) -> Any:
        """Auto-parse JSON-string lists before Pydantic validation.

        MiniMax frequently returns list[str] fields as JSON strings like
        '["item1", "item2"]'. This validator catches ALL such cases at
        the schema level, regardless of nesting or special characters.
        """
        import re

        if not isinstance(data, dict):
            return data
        schema = cls.model_json_schema()
        props = schema.get("properties", {})
        for key, prop in props.items():
            if prop.get("type") == "array" and isinstance(data.get(key), str):
                raw = data[key].strip()
                if raw.startswith("["):
                    # Try as-is
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            data[key] = parsed
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    # Escape unescaped control chars
                    cleaned = re.sub(
                        r'(?<!\\)([\n\r\t])',
                        lambda m: {"\n": "\\n", "\r": "\\r", "\t": "\\t"}[m.group(1)],
                        raw,
                    )
                    try:
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, list):
                            data[key] = parsed
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                    # Last resort: split by common JSON array pattern
                    try:
                        # Strip outer brackets, split by '", "' pattern
                        inner = raw[1:-1].strip()
                        if inner.startswith('"') and inner.endswith('"'):
                            items = [s.strip().strip('"') for s in inner.split('",')]
                            items = [s.strip('"') for s in items]
                            data[key] = items
                    except Exception:
                        pass
        return data
