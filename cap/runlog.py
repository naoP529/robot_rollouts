"""ロールアウトの記録。方針の生成ループからは RunLog のメソッドだけを呼ぶ。"""

from __future__ import annotations

import functools
import inspect
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cap.blocks.drive import pose_from_env

PERCEPTION = {"detect", "search", "nudge"}
RUNS = Path(__file__).resolve().parent / "runs"
MAX_TRANSCRIPT_LINES = 1000


class AttemptTimeout(Exception):
    """Raised when a single attempt's wall-clock budget runs out before an API call.

    Generated policy code can loop (e.g. a reacquire-the-ball search that never
    converges) without ever raising on its own, so nothing else would stop it short of
    this. It is meant to be caught exactly like any other attempt failure, not retried
    internally.
    """


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.attempt = 0
        self.events: list[dict] = []
        self.policies: dict[int, str] = {}
        self.helpers: dict[int, dict[str, str]] = {}
        self.meta: dict = {}
        self._i = 0
        self._closed = False

    @classmethod
    def create(cls, seed: int) -> "RunLog":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUNS / f"{stamp}_s{seed}"
        n = 2
        while path.exists():
            path = RUNS / f"{stamp}_s{seed}_{n}"
            n += 1
        path.mkdir(parents=True)
        return cls(path)

    @classmethod
    def from_dir(cls, path: Path) -> "RunLog":
        path = Path(path)
        log = cls(path)
        meta_path = path / "meta.json"
        if meta_path.is_file():
            log.meta = json.loads(meta_path.read_text())
        events_path = path / "events.jsonl"
        if events_path.is_file():
            with events_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        log.events.append(json.loads(line))
        if log.events:
            log._i = int(log.events[-1].get("i", len(log.events) - 1)) + 1
        log._closed = True
        code_root = path / "code"
        if code_root.is_dir():
            for attempt_dir in sorted(code_root.iterdir()):
                if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
                    continue
                attempt = int(attempt_dir.name.split("_", 1)[1])
                policy = attempt_dir / "policy.py"
                if policy.is_file():
                    log.policies[attempt] = policy.read_text()
                helpers_dir = attempt_dir / "helpers"
                if helpers_dir.is_dir():
                    for helper in sorted(helpers_dir.glob("*.py")):
                        log.helpers.setdefault(attempt, {})[helper.stem] = helper.read_text()
        return log

    def start(self, meta: dict, env) -> None:
        self.meta = {
            "id": self.path.name,
            **meta,
            "started_at": _now(),
            "ended_at": None,
            "success": None,
            "stop_reason": None,
            "skills_read": [],
        }
        self._write_meta()
        self._append({"kind": "start", "truth": truth_snapshot(env)})

    def set_attempt(self, attempt: int) -> None:
        self.attempt = attempt

    def note_skills(self, names: list[str]) -> None:
        have = self.meta.setdefault("skills_read", [])
        for name in names:
            if name not in have:
                have.append(name)
        self._write_meta()

    def save_policy(self, code: str) -> None:
        dest = self.path / "code" / f"attempt_{self.attempt}" / "policy.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_with_newline(code))
        self.policies[self.attempt] = code

    def save_helper(self, name: str, code: str) -> None:
        if not str(name).isidentifier():
            return
        dest = self.path / "code" / f"attempt_{self.attempt}" / "helpers" / f"{name}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_with_newline(code))
        self.helpers.setdefault(self.attempt, {})[name] = code

    def api(self, name: str, args: dict, result, error: str | None, wrapper) -> None:
        # belief/truth are only ever read for PERCEPTION calls (see _error_lines below),
        # so skip the state dump and sim readout for every other API call.
        track = name in PERCEPTION
        result_j = _jsonable(result)
        ids = _tracked_ids(args, result_j) if track else None
        self._append(
            {
                "attempt": self.attempt,
                "kind": "api",
                "name": name,
                "args": args,
                "result": result_j,
                "error": error,
                "belief": _belief_snapshot(wrapper.state, ids) if track else None,
                "truth": truth_snapshot(wrapper.env) if track else None,
            }
        )

    def exception(self, exc: BaseException) -> None:
        tail = _tail("".join(traceback.format_exception(exc)))
        self._append({"attempt": self.attempt, "kind": "exception", "error": tail})

    def finish(self, success: bool, stop_reason: str, env) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        self._append(
            {
                "kind": "end",
                "success": success,
                "stop_reason": stop_reason,
                "truth": truth_snapshot(env),
            }
        )
        self.meta["success"] = success
        self.meta["stop_reason"] = stop_reason
        self.meta["ended_at"] = _now()
        started = self.meta.get("started_at")
        self.meta["duration_s"] = (
            (datetime.fromisoformat(self.meta["ended_at"]) - datetime.fromisoformat(started)).total_seconds()
            if started
            else None
        )
        self._write_meta()
        self.write_transcript()
        return self.path

    def write_transcript(self) -> Path:
        dest = self.path / "transcript.md"
        dest.write_text(_transcript(self))
        return dest

    def _append(self, event: dict) -> None:
        row = {"i": self._i, **event}
        self._i += 1
        self.events.append(row)
        with (self.path / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def _write_meta(self) -> None:
        (self.path / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")


def logged_namespace(wrapper, log: RunLog, timeout_s: float | None = None) -> dict:
    """方針に渡す API。呼び出しのあとで信念と実位置を1行残す。

    timeout_s を渡すと、この名前空間を作った時点からの経過時間がそれを超えた直後の
    API呼び出しで AttemptTimeout を送出し、無限ループする生成コードを打ち切る。
    """
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    return {name: _wrap_api(name, fn, wrapper, log, deadline) for name, fn in wrapper.namespace().items()}


def truth_snapshot(env) -> dict:
    balls = []
    entities = getattr(env, "_ball_entities", None) or []
    for spec, entity in zip(getattr(env, "balls_spec", []), entities):
        x, y = _xy(entity.get_pos())
        balls.append({"name": spec["name"], "x": x, "y": y})
    arches = []
    specs = getattr(env, "arches_spec", None)
    ys = getattr(env, "arch_ys", None)
    xs = getattr(env, "arch_xs", None)
    if specs is not None and ys is not None and xs is not None:
        for spec, x, y in zip(specs, xs, ys):
            arches.append({"name": spec["name"], "x": float(x), "y": float(y)})
    return {"robot": pose_from_env(env).to_dict(), "balls": balls, "arches": arches}


def _wrap_api(name: str, fn, wrapper, log: RunLog, deadline: float | None):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if deadline is not None and time.monotonic() > deadline:
            exc = AttemptTimeout(f"attempt exceeded its time budget before calling {name}()")
            log.api(name, {}, None, f"{type(exc).__name__}: {exc}", wrapper)
            raise exc
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        payload = {key: _jsonable(value) for key, value in bound.arguments.items()}
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            log.api(name, payload, None, f"{type(exc).__name__}: {exc}", wrapper)
            raise
        log.api(name, payload, result, None, wrapper)
        return result

    return inner


def _xy(pos) -> tuple[float, float]:
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    row = np.asarray(pos).reshape(-1, 3)
    return float(row[0, 0]), float(row[0, 1])


def _jsonable(value):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _tracked_ids(args: dict | None, result) -> list[str] | None:
    if isinstance(result, list) and all(isinstance(item, str) for item in result):
        return result
    obj_id = (args or {}).get("obj_id")
    if isinstance(obj_id, str):
        return [obj_id]
    return None


def _belief_snapshot(state, ids: list[str] | None) -> dict:
    observed = state.observed
    if ids is not None:
        want = set(ids)
        observed = [item for item in observed if item.id in want]
    return {
        "robot": {"pose": state.robot.pose.to_dict()},
        "observed": [item.to_dict() for item in observed],
    }


def _transcript(log: RunLog) -> str:
    detailed = _render_transcript(log, summarized=False)
    if detailed.count("\n") + 1 <= MAX_TRANSCRIPT_LINES:
        return detailed
    return _render_transcript(log, summarized=True)


def _render_transcript(log: RunLog, *, summarized: bool) -> str:
    lines = [
        f"# Rollout {log.meta.get('id', log.path.name)}",
        "",
        f"Task: {str(log.meta.get('task', '')).strip()}",
        f"Success: {log.meta.get('success')}",
        f"Stop: {log.meta.get('stop_reason')}",
        f"Duration: {_fmt_duration(log.meta.get('duration_s'))}",
        "",
    ]
    start = next((event for event in log.events if event["kind"] == "start"), None)
    end = next((event for event in reversed(log.events) if event["kind"] == "end"), None)
    if start is not None:
        lines.append("## Initial truth")
        lines.extend(_truth_lines(start["truth"]))
        lines.append("")
    attempts = sorted(
        set(log.policies)
        | set(log.helpers)
        | {event["attempt"] for event in log.events if "attempt" in event}
    )
    for attempt in attempts:
        lines.append(f"## Attempt {attempt}")
        lines.append("")
        code = log.policies.get(attempt)
        if code:
            lines.extend(_fence(code))
            lines.append("")
        for name, helper in log.helpers.get(attempt, {}).items():
            lines.append(f"Helper `{name}`:")
            lines.append("")
            lines.extend(_fence(helper))
            lines.append("")
        attempt_events = [event for event in log.events if event.get("attempt") == attempt]
        if summarized:
            lines.extend(_summarize_attempt(attempt_events, log.meta))
        else:
            lines.extend(_detail_attempt(attempt_events))
        lines.append("")
    if end is not None:
        lines.append("## Final truth")
        lines.extend(_truth_lines(end["truth"]))
        lines.append("")
    return "\n".join(lines)


def _detail_attempt(events: list[dict]) -> list[str]:
    lines = []
    number = 0
    for event in events:
        if event["kind"] == "api":
            number += 1
            lines.append(f"{number}. {_fmt_call(event['name'], event['args'])} -> {_fmt_result(event)}")
            if event["name"] in PERCEPTION and not event.get("error"):
                lines.extend(_error_lines(event))
        elif event["kind"] == "exception":
            lines.extend(_exception_lines(event))
    return lines


def _summarize_attempt(events: list[dict], meta: dict) -> list[str]:
    by_name: dict[str, list[dict]] = {}
    for event in events:
        if event["kind"] == "api":
            by_name.setdefault(event["name"], []).append(event)
    lines: list[str] = []
    if look_at := by_name.get("look_at"):
        lines.append(f"- look_at ×{len(look_at)}")
    if finished := by_name.get("is_finished"):
        lines.extend(_summarize_is_finished(finished))
    for name in ("detect", "search"):
        if name in by_name:
            lines.extend(_summarize_perception(name, by_name[name]))
    if gotos := by_name.get("goto"):
        lines.extend(_summarize_goto(gotos))
    if nudges := by_name.get("nudge"):
        lines.extend(_summarize_nudge(nudges, meta))
    for event in events:
        if event["kind"] == "exception":
            lines.extend(_exception_lines(event))
    return lines


def _summarize_is_finished(events: list[dict]) -> list[str]:
    trues = [event for event in events if event.get("result") is True]
    lines = []
    if trues:
        for event in trues:
            lines.append(f"- {_fmt_call(event['name'], event['args'])} -> true")
        n_false = len(events) - len(trues)
        if n_false:
            lines.append(f"- is_finished ×{n_false} -> false")
        return lines
    return [f"- is_finished ×{len(events)} -> false"]


def _summarize_perception(name: str, events: list[dict]) -> list[str]:
    hits = []
    errors: list[tuple[str, str, float]] = []
    for event in events:
        result = event.get("result")
        hits.append(len(result) if isinstance(result, list) else 0)
        errors.extend(_pose_errors(event))
    lines = [f"- {name} ×{len(events)}"]
    if hits:
        lines.append(f"  ids per call: {min(hits)}–{max(hits)}")
    if errors:
        closest = min(errors, key=lambda row: row[2])
        farthest = max(errors, key=lambda row: row[2])
        lines.append(f"  closest: {closest[0]} {closest[2]:.2f} m from {closest[1]}")
        lines.append(f"  farthest: {farthest[0]} {farthest[2]:.2f} m from {farthest[1]}")
    return lines


def _summarize_goto(events: list[dict]) -> list[str]:
    first = _fmt_xyyaw(events[0].get("args") or {})
    last = _fmt_xyyaw(events[-1].get("args") or {})
    lines = [f"- goto ×{len(events)}"]
    if first:
        lines.append(f"  first: {first}")
    if last and len(events) > 1:
        lines.append(f"  last: {last}")
    return lines


def _summarize_nudge(events: list[dict], meta: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for event in events:
        obj_id = (event.get("args") or {}).get("obj_id")
        if isinstance(obj_id, str) and obj_id not in seen:
            seen.add(obj_id)
            ids.append(obj_id)
    lines = [f"- nudge ×{len(events)}"]
    if ids:
        lines.append(f"  ids: {', '.join(ids)}")
    target = meta.get("target_ball")
    start = _named_xy(events[0].get("truth") or {}, target) if target else None
    end = _named_xy(events[-1].get("truth") or {}, target) if target else None
    if start and end:
        moved = math.hypot(end[0] - start[0], end[1] - start[1]) > 1e-3
        lines.append(
            f"  {target} truth: ({start[0]:.2f}, {start[1]:.2f}) -> ({end[0]:.2f}, {end[1]:.2f}) "
            f"moved: {str(moved).lower()}"
        )
    return lines


def _exception_lines(event: dict) -> list[str]:
    return [
        "Exception:",
        "",
        "```",
        str(event.get("error") or "").rstrip("\n"),
        "```",
    ]


def _truth_lines(truth: dict) -> list[str]:
    robot = truth["robot"]
    yaw = robot.get("yaw")
    yaw_text = f" yaw={yaw:.2f}" if isinstance(yaw, (int, float)) else ""
    lines = [f"- robot: x={robot['x']:.2f} y={robot['y']:.2f}{yaw_text}"]
    for item in [*truth["balls"], *truth["arches"]]:
        lines.append(f"- {item['name']}: x={item['x']:.2f} y={item['y']:.2f}")
    return lines


def _error_lines(event: dict) -> list[str]:
    lines = []
    for obj_id, name, error, pose, other in _pose_errors(event, with_xy=True):
        lines.append(f"   belief {obj_id}: ({pose[0]:.2f}, {pose[1]:.2f})")
        lines.append(f"   truth {name}: ({other[0]:.2f}, {other[1]:.2f})")
        lines.append(f"   error: {error:.2f} m")
    return lines


def _pose_errors(event: dict, *, with_xy: bool = False):
    belief = event.get("belief") or {}
    truth = event.get("truth") or {}
    real = {item["name"]: item for item in [*truth.get("balls", []), *truth.get("arches", [])]}
    ids = _tracked_ids(event.get("args"), event.get("result"))
    rows = []
    for item in belief.get("observed", []):
        if ids is not None and item.get("id") not in ids:
            continue
        pose = item.get("pose") or {}
        other = real.get(item.get("name"))
        if other is None or "x" not in pose or "y" not in pose:
            continue
        error = math.hypot(pose["x"] - other["x"], pose["y"] - other["y"])
        if with_xy:
            rows.append((item["id"], item["name"], error, (pose["x"], pose["y"]), (other["x"], other["y"])))
        else:
            rows.append((item["id"], item["name"], error))
    return rows


def _named_xy(truth: dict, name: str | None) -> tuple[float, float] | None:
    if not name:
        return None
    for item in [*truth.get("balls", []), *truth.get("arches", [])]:
        if item.get("name") == name:
            return float(item["x"]), float(item["y"])
    return None


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m{rest:04.1f}s"


def _fmt_xyyaw(args: dict) -> str:
    parts = []
    for key in ("x", "y", "yaw"):
        value = args.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.2f}")
        elif key in args:
            parts.append(f"{key}={value!r}")
    return " ".join(parts)


def _fmt_call(name: str, args: dict) -> str:
    parts = [f"{key}={value!r}" for key, value in args.items()]
    return f"{name}({', '.join(parts)})"


def _fmt_result(event: dict) -> str:
    if event.get("error"):
        return f"error: {event['error']}"
    return json.dumps(event["result"])


def _fence(code: str) -> list[str]:
    return ["```python", code.rstrip("\n"), "```"]


def _with_newline(code: str) -> str:
    return code if code.endswith("\n") else code + "\n"


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
