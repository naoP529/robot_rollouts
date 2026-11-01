"""ロールアウト終了後のスキル更新。transcript はプロンプトに埋め、ツールにはしない。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from cap.usage import record_call
from cap.utils.prompts import REVIEW_PROMPT
from cap.utils.skill_tools import (
    TOOLS,
    WRITE_TOOLS,
    create_skill,
    read_skills,
    replace_in_skill_file,
    skill_list,
    write_skill_file,
)

MAX_ROUNDS = 8
REVIEW_TOOLS = [TOOLS[0], *WRITE_TOOLS]
REVIEW_FNS = {
    "read_skills": read_skills,
    "create_skill": create_skill,
    "write_skill_file": write_skill_file,
    "replace_in_skill_file": replace_in_skill_file,
}

# Rate limits and transient 5xx/connection errors resolve on their own; retrying with
# backoff turns a momentary API hiccup into a short wait instead of a crashed process or
# a silently dropped review of an otherwise-successful rollout.
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def create_with_retry(client, *, max_retries: int = 5, base_delay: float = 5.0, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"[retry] {type(exc).__name__} from the LLM API, retrying in {delay:.0f}s ({attempt + 1}/{max_retries})")
            time.sleep(delay)


def review_rollout(llm, run_dir: Path) -> None:
    path = Path(run_dir)
    transcript_path = path / "transcript.md"
    if not transcript_path.is_file():
        return
    prompt = REVIEW_PROMPT.replace("{skill_list}", skill_list() or "(none)", 1)
    prompt = prompt.replace("{transcript}", transcript_path.read_text(), 1)
    messages: list = [{"role": "user", "content": prompt}]
    final = ""
    for _ in range(MAX_ROUNDS):
        response = create_with_retry(
            llm.client,
            model=llm.model_name,
            messages=messages,
            temperature=llm.temperature,
            reasoning_effort=llm.effort,
            tools=REVIEW_TOOLS,
        )
        record_call(llm, response, phase="review")
        message = response.choices[0].message
        if not message.tool_calls:
            final = message.content or ""
            break
        messages.append(message)
        for call in message.tool_calls:
            result = _call_tool(path, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        final = "Stopped: tool round cap reached."
    (path / "review.md").write_text(final if final.endswith("\n") else final + "\n")


def _call_tool(run_dir: Path, call) -> str:
    name = call.function.name
    raw = call.function.arguments or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        _append(run_dir, {"name": name, "args": raw, "result": f"error: {exc}"})
        return f"error: {exc}"
    fn = REVIEW_FNS.get(name)
    try:
        result = f"error: unknown tool {name}" if fn is None else str(fn(**args))
    except Exception as exc:
        result = f"error: {exc}"
    _append(run_dir, {"name": name, "args": args, "result": result})
    return result


def _append(run_dir: Path, row: dict) -> None:
    with (run_dir / "review.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
