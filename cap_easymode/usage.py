"""ロールアウトごとの LLM トークンとキャッシュの記録。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

USAGES = Path(__file__).resolve().parent / "usages"


class UsageLog:
    def __init__(self, path: Path, run_id: str, model: str, effort: str, temperature: float):
        self.path = path
        self.run_id = run_id
        self.model = model
        self.effort = effort
        self.temperature = temperature
        self.calls: list[dict] = []

    @classmethod
    def create(cls, run_id: str, *, model: str, effort: str, temperature: float) -> "UsageLog":
        USAGES.mkdir(parents=True, exist_ok=True)
        log = cls(USAGES / f"{run_id}.json", run_id, model, effort, temperature)
        log._write()
        return log

    def record(
        self,
        response,
        *,
        phase: str,
        attempt: int | None = None,
        helper: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
    ) -> None:
        model = self.model if model is None else model
        effort = self.effort if effort is None else effort
        temperature = self.temperature if temperature is None else temperature
        usage = _usage_of(response)
        tokens = _tokens(usage)
        cached = tokens["cached_tokens"]
        prompt = tokens["prompt_tokens"]
        row = {
            "i": len(self.calls),
            "phase": phase,
            "at": _now(),
            "model": model,
            "effort": effort,
            "temperature": temperature,
            **tokens,
            "cache_hit": cached > 0,
            "cache_ratio": None if not prompt else cached / prompt,
            "usage": usage,
        }
        if phase in {"policy", "helper"} and attempt is not None:
            row["attempt"] = attempt
        if helper is not None:
            row["helper"] = helper
        returned = getattr(response, "model", None)
        if returned and returned != model:
            row["response_model"] = returned
        self.calls.append(_order(row))
        self._write()

    def _write(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "id": self.run_id,
                    "model": self.model,
                    "effort": self.effort,
                    "temperature": self.temperature,
                    "calls": self.calls,
                    "totals": _totals(self.calls),
                },
                indent=2,
            )
            + "\n"
        )


def record_call(llm, response, **kwargs) -> None:
    log = getattr(llm, "usage", None)
    if log is None:
        return
    log.record(
        response,
        model=llm.model_name,
        effort=llm.effort,
        temperature=llm.temperature,
        **kwargs,
    )


def _usage_of(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
    return _dump(usage)


def _dump(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _dump(item) if _nested(item) else item for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except TypeError:
            return value.model_dump(mode="json")
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    return value


def _nested(value) -> bool:
    return isinstance(value, dict) or hasattr(value, "model_dump") or (
        hasattr(value, "dict") and callable(getattr(value, "dict", None))
    )


def _tokens(usage: dict | None) -> dict:
    if not usage:
        return {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    prompt = _int(_get(usage, "prompt_tokens", "promptTokenCount"))
    details = _as_dict(_get(usage, "prompt_tokens_details", "promptTokensDetails"))
    cached = _get(details, "cached_tokens")
    if cached is None:
        cached = _get(usage, "cached_tokens", "cached_content_token_count", "cachedContentTokenCount")
    cached = _int(cached)
    completion = _int(_get(usage, "completion_tokens", "candidatesTokenCount"))
    completion_details = _as_dict(_get(usage, "completion_tokens_details", "completionTokensDetails"))
    reasoning = _get(completion_details, "reasoning_tokens")
    if reasoning is None:
        reasoning = _get(usage, "reasoning_tokens", "thoughts_token_count", "thoughtsTokenCount")
    reasoning = _int(reasoning)
    total = _get(usage, "total_tokens", "totalTokenCount")
    total = prompt + completion if total is None else _int(total)
    cached = min(cached, prompt)
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "uncached_prompt_tokens": max(prompt - cached, 0),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def _totals(calls: list[dict]) -> dict:
    keys = (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    totals = {key: sum(int(call.get(key) or 0) for call in calls) for key in keys}
    totals["calls"] = len(calls)
    totals["cache_hits"] = sum(1 for call in calls if call.get("cache_hit"))
    return {
        "calls": totals["calls"],
        "cache_hits": totals["cache_hits"],
        **{key: totals[key] for key in keys},
    }


def _order(row: dict) -> dict:
    keys = ["i", "phase"]
    if "attempt" in row:
        keys.append("attempt")
    if "helper" in row:
        keys.append("helper")
    keys.extend(
        [
            "at",
            "model",
            "effort",
            "temperature",
            "prompt_tokens",
            "cached_tokens",
            "uncached_prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cache_hit",
            "cache_ratio",
        ]
    )
    if "response_model" in row:
        keys.append("response_model")
    keys.append("usage")
    return {key: row[key] for key in keys if key in row}


def _get(data: dict | None, *names):
    if not data:
        return None
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _as_dict(value) -> dict:
    dumped = _dump(value)
    return dumped if isinstance(dumped, dict) else {}


def _int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
