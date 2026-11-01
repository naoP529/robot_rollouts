from dotenv import load_dotenv
from openai import OpenAI
import builtins, os, json, argparse, inspect, random

from dataclasses import dataclass
from pathlib import Path

from cap_easymode.review import create_with_retry, review_rollout
from cap_easymode.runlog import AttemptTimeout, RunLog, logged_namespace
from cap_easymode.usage import UsageLog, record_call
from cap_easymode.utils.skill_tools import (
    exec_skill_scripts,
    read_skills,
    skill_list,
    TOOLS,
)
from cap_easymode.utils.prompts import SYSTEM_PROMPTS, GLOBAL_SETTINGS, TASKS, SKILLS, LMP_PROMPT, LMP_F_PROMPT
from cap_easymode.utils.wrapper import DEFAULT_TEXTURE, Wrapper
from envs.cfg.model_data import ARCHES, BALLS

load_dotenv()

# OpenAI SDK経由で、OpenAI互換のエンドポイントなら何にでも向けられるようにしている。
# 未設定ならこれまで通りGeminiをデフォルトにする。他プロバイダに切り替えるには.envで
# LLM_BASE_URL/LLM_MODEL_NAMEを上書きするだけでよい(gemini_api_keyは後方互換のため残す)。
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("gemini_api_key")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MAX_ATTEMPTS = 3
# Wall-clock budget per attempt. This is a pass/fail criterion, not just an infinite-loop
# safety net: an attempt that would eventually finish but takes too long is still a
# failure, the same as AttemptTimeout cutting off a genuinely stuck loop. Skills that make
# the robot slow but more careful (more staging waypoints, more re-verification) trade
# directly against this budget, so review should weigh speed against success, not just
# chase success. Observed attempts without this pressure ranged from ~2.5 min to 74 min.
DEFAULT_MAX_SECONDS = 300.0

DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-4-argon")
DEFAULT_EFFORT="medium"
DEFAULT_TEMPRATURE=1.0

TOOL_FNS = {
    "read_skills": read_skills,
    "exec_skill_scripts": exec_skill_scripts,
}

def build_api_doc(
    namespace:dict
) -> str:
    lines = []
    for name, fn in namespace.items():
        sig = inspect.signature(fn) #引数の形
        doc = (fn.__doc__ or "").strip()
        lines.append(f"{name}{sig}: {doc}")
    return "\n".join(lines)

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

@dataclass
class LLMSettings:
    client:OpenAI
    model_name:str
    effort:str
    temperature:float
    usage: UsageLog | None = None

def generate_policy_code(
    llm:LLMSettings,
    task: str,
    settings: str,
    system: str,
    skills: str,
    apis: str,
    *,
    attempt: int,
) -> tuple[str, list[str]]:
    prompts = LMP_PROMPT.format(
        system=system, settings=settings, task=task, skills=skills.format(skill_list=skill_list()), apis=apis
    )
    messages = [{"role":"user", "content":prompts}]
    skills_read: list[str] = []

    while True:
        response = create_with_retry(
            llm.client,
            model=llm.model_name,
            messages=messages,
            temperature=llm.temperature,
            reasoning_effort=llm.effort,
            tools=TOOLS
        )
        record_call(llm, response, phase="policy", attempt=attempt)
        message = response.choices[0].message
        if not message.tool_calls:
            return _strip_code_fence(message.content or ""), skills_read

        messages.append(message)
        for call in message.tool_calls:
            fn = TOOL_FNS[call.function.name]
            args = json.loads(call.function.arguments)
            skill_name = args.get("skill_name")
            if call.function.name == "read_skills" and skill_name and skill_name not in skills_read:
                skills_read.append(skill_name)
            try:
                result = fn(**args)
            except Exception as e:
                result = f"error: {e}"

            messages.append(
                {"role":"tool", "tool_call_id":call.id, "content":str(result)}
            )

def generate_sub_function(
    llm:LLMSettings, name: str, args: tuple, kwargs: dict, system:str, settings:str, apis:str, *, attempt: int
) -> str:
    call_args = ", ".join([*(repr(a) for a in args), *(f"{k}={v!r}" for k, v in kwargs.items())])
    prompt = LMP_F_PROMPT.format(
        system=system, settings=settings, apis=apis, signature=f"{name}({call_args})"
    )
    response = create_with_retry(
        llm.client,
        model=llm.model_name,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=llm.effort,
        temperature=llm.temperature
    )
    record_call(llm, response, phase="helper", attempt=attempt, helper=name)
    return _strip_code_fence(response.choices[0].message.content or "")

class LazyNamespace(dict):
    def __init__(self, base_namespace:dict, apis:str, system:str, settings:str, llm:LLMSettings, on_helper=None, attempt: int = 0):
        super().__init__(base_namespace)
        self._llm = llm
        self._apis = apis
        self._system = system
        self._settings = settings
        self._on_helper = on_helper
        self._attempt = attempt

    def __missing__(self, key: str):
        if hasattr(builtins, key):
            raise KeyError(key)
        def _lazy(*args, **kwargs):
            code = generate_sub_function(
                self._llm, key, args, kwargs, self._system, self._settings, self._apis, attempt=self._attempt
            )
            if self._on_helper is not None:
                self._on_helper(key, code)
            exec(compile(code, f"<lmp_f:{key}>", "exec"), self)  # self(=この名前空間)に関数を定義
            real = self.get(key)
            if real is None or real is _lazy:
                raise NameError(f"LMP failed to define: {key}")
            return real(*args, **kwargs)

        self[key] = _lazy  # 同じ回の実行内で2回呼ばれても再生成しない
        return _lazy

def run_policy(llm:LLMSettings, code: str, base_namespace: dict, apis: str, system: str, settings: str, on_helper=None, attempt: int = 0):
    ns = LazyNamespace(base_namespace, apis, system, settings, llm, on_helper, attempt)
    exec(compile(code, "<lmp_policy>", "exec"), ns)

DEFAULT_SCENE_SIZE = 4

def _asset_specs(
    names: list[str] | None,
    catalog: list[dict],
    *,
    include: str | None = None,
    count: int = DEFAULT_SCENE_SIZE,
    seed: int = 0,
) -> list[dict]:
    found = {item["name"]: item for item in catalog}
    if names:
        missing = [name for name in names if name not in found]
        if missing:
            raise ValueError(f"unknown asset: {', '.join(missing)}")
        return [dict(found[name]) for name in names]
    if include is not None and include not in found:
        raise ValueError(f"unknown asset: {include}")
    pool = [item["name"] for item in catalog if item["name"] != include]
    random.Random(seed).shuffle(pool)
    room = max(count - (1 if include is not None else 0), 0)
    chosen = ([include] if include is not None else []) + pool[:room]
    return [dict(found[name]) for name in chosen]

def _color(name: str, kind: str) -> str:
    prefix = f"{kind}_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name

def _rollout_settings(ball_specs: list[dict], arch_specs: list[dict]) -> str:
    names = ", ".join(spec["name"] for spec in (*ball_specs, *arch_specs))
    return (
        GLOBAL_SETTINGS.rstrip()
        + "\n- Pass asset names, not color words, to get_loc, nudge, and is_finished."
        + " Names are ball_<color> and arch_<color>."
        + f"\n- Assets in this rollout: {names}\n"
    )

def rollouts(
    model:str = DEFAULT_MODEL_NAME,
    effort:str = DEFAULT_EFFORT,
    temperature:float = DEFAULT_TEMPRATURE,
    balls:list[str] | None = None,
    arches:list[str] | None = None,
    lighting:str = "",
    texture:str = "",
    show_viewer:bool = True,
    record_video:bool = True,
    max_attempts:int = MAX_ATTEMPTS,
    max_seconds:float | None = DEFAULT_MAX_SECONDS,
    seed:int = 0,
    *,
    target_ball:str,
    target_arch:str,
) -> Path:
    ball_specs = _asset_specs(balls, BALLS, include=target_ball, seed=seed)
    arch_specs = _asset_specs(arches, ARCHES, include=target_arch, seed=seed)
    task = TASKS.format(ball_color=_color(target_ball, "ball"), arch_color=_color(target_arch, "arch")).strip()
    settings = _rollout_settings(ball_specs, arch_specs)
    wrapper = Wrapper(
        balls=ball_specs,
        arches=arch_specs,
        lighting=lighting or "clear_noon",
        texture=texture or None,
        seed=seed,
        show_viewer=show_viewer,
    )
    log = RunLog.create(seed)
    video_path = log.path / "rollout.mp4"
    if record_video:
        # head camera only: the static overview cameras currently render blank on this machine.
        wrapper.env.camera.start_recording(str(video_path), fps=30)
    llm = LLMSettings(
        client=OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL),
        model_name=model,
        effort=effort,
        temperature=temperature,
        usage=UsageLog.create(log.path.name, model=model, effort=effort, temperature=temperature),
    )
    success = False
    stop_reason = "max_attempts"
    try:
        log.start(
            {
                "model": model,
                "effort": effort,
                "temperature": temperature,
                "seed": seed,
                "task": task,
                "target_ball": target_ball,
                "target_arch": target_arch,
                "video": video_path.name if record_video else None,
                "scene": {
                    "balls": [spec["name"] for spec in ball_specs],
                    "arches": [spec["name"] for spec in arch_specs],
                    "lighting": lighting or "clear_noon",
                    "texture": texture or DEFAULT_TEXTURE,
                },
            },
            wrapper.env,
        )
        for attempt in range(max_attempts):
            log.set_attempt(attempt)
            try:
                apis = build_api_doc(wrapper.namespace())
                code, skills = generate_policy_code(
                    llm, task, settings, SYSTEM_PROMPTS, SKILLS, apis, attempt=attempt
                )
                log.note_skills(skills)
                log.save_policy(code)
                run_policy(
                    llm,
                    code,
                    logged_namespace(wrapper, log, timeout_s=max_seconds),
                    apis,
                    SYSTEM_PROMPTS,
                    settings,
                    log.save_helper,
                    attempt,
                )
                if wrapper.is_finished(target_ball, target_arch):
                    success = True
                    stop_reason = "finished"
                    break
            except Exception as exc:
                log.exception(exc)
                if attempt == max_attempts - 1:
                    stop_reason = "timeout" if isinstance(exc, AttemptTimeout) else "exception"
    except Exception:
        if not success:
            stop_reason = "exception"
        raise
    finally:
        if record_video:
            try:
                wrapper.env.camera.stop_recording()
            except Exception:
                pass
        log.finish(success, stop_reason, wrapper.env)
    try:
        review_rollout(llm, log.path)
    except Exception as exc:
        # The rollout itself is already finished and logged at this point; a review
        # failure (e.g. a rate limit that outlasted the retries) shouldn't take that
        # down with it. Skills just don't get updated for this run.
        print(f"[warn] review_rollout failed, skills not updated for this run: {exc}")
    return log.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPRATURE)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Wall-clock budget per attempt; 0 disables the timeout.",
    )
    parser.add_argument("--target-ball", required=True)
    parser.add_argument("--target-arch", required=True)
    parser.add_argument("--balls", nargs="*", default=None)
    parser.add_argument("--arches", nargs="*", default=None)
    parser.add_argument("--lighting", default="")
    parser.add_argument("--texture", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rollouts(
        model=args.model,
        effort=args.effort,
        temperature=args.temperature,
        balls=args.balls,
        arches=args.arches,
        lighting=args.lighting,
        texture=args.texture,
        show_viewer=not args.no_viewer,
        record_video=not args.no_video,
        max_attempts=args.max_attempts,
        max_seconds=args.max_seconds or None,
        seed=args.seed,
        target_ball=args.target_ball,
        target_arch=args.target_arch,
    )


if __name__ == "__main__":
    main()
