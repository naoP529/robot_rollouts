"""Run one real rollout and film it from several cameras at once, for footage rather
than evaluation: the dog's first-person head camera, plus a few bird's-eye angles, each
written to its own mp4 in the run's log directory.

A separate file from rollout.py / cap(_easymode)/main.py on purpose: this duplicates the
small slice of cap_easymode.main.rollouts()'s orchestration needed to start/stop recording
on multiple cameras (ArcadeEnv's scene.step() feeds every camera with an active recording
automatically, see envs/arcade.py's camera setup), instead of changing those files.

Defaults to cap_easymode (exact get_loc, no owl/command2loc) because it actually finishes
rollouts; pass --no-easymode to film the real cap/ pipeline instead (expect more failures).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from rollout import PRESETS

# Picked out of envs/cfg/lights.py's full preset list for visual drama. snowfield is
# excluded: its pale ambient_light/background_color wash everything out to flat grey
# regardless of ground texture, which reads as low-contrast/hard to see, not cinematic.
CINEMATIC_LIGHTING = [
    "golden_hour",
    "sunset",
    "moonlight",
    "studio",
    "street_lamp_night",
]

# All ground textures under assets/texture/, each a diffuse/normal/roughness.jpg set.
CINEMATIC_TEXTURES = [
    "leafy_glass",
    "brick_pavement",
    "clean_asphalt",
    "gravel_floor",
    "snow",
    "walnut_veneer",
]

# Camera names ArcadeEnv always builds (see envs/arcade.py's _overview_cameras): "head" is
# the dog's first-person view; "top" looks straight down; the "_left"/"_right" pairs are
# 35-degree elevated corners. Recording more cameras means more video files and disk use.
ALL_CAMERAS = ["head", "top", "front_left", "front_right", "back_left", "back_right"]
DEFAULT_CAMERAS = ["head", "top"]


def _select_backend(easymode: bool):
    if easymode:
        import cap_easymode.main as main_mod
        import cap_easymode.review as review_mod
        import cap_easymode.runlog as runlog_mod
        import cap_easymode.usage as usage_mod
        import cap_easymode.utils.prompts as prompts_mod
        import cap_easymode.utils.wrapper as wrapper_mod
    else:
        import cap.main as main_mod
        import cap.review as review_mod
        import cap.runlog as runlog_mod
        import cap.usage as usage_mod
        import cap.utils.prompts as prompts_mod
        import cap.utils.wrapper as wrapper_mod
    return main_mod, review_mod, runlog_mod, usage_mod, prompts_mod, wrapper_mod


@dataclass
class LLMSettings:
    client: OpenAI
    model_name: str
    effort: str
    temperature: float
    usage: object = None


def film_rollout(
    *,
    target_ball: str,
    target_arch: str,
    balls: list[str] | None = None,
    arches: list[str] | None = None,
    lighting: str,
    texture: str,
    cameras: list[str],
    fps: int,
    easymode: bool,
    model: str,
    effort: str,
    temperature: float,
    max_attempts: int,
    max_seconds: float | None,
    show_viewer: bool,
    seed: int,
) -> Path:
    main_mod, review_mod, runlog_mod, usage_mod, prompts_mod, wrapper_mod = _select_backend(easymode)

    ball_specs = main_mod._asset_specs(balls, main_mod.BALLS, include=target_ball, seed=seed)
    arch_specs = main_mod._asset_specs(arches, main_mod.ARCHES, include=target_arch, seed=seed)
    task = prompts_mod.TASKS.format(
        ball_color=main_mod._color(target_ball, "ball"), arch_color=main_mod._color(target_arch, "arch")
    ).strip()
    settings = main_mod._rollout_settings(ball_specs, arch_specs)

    wrapper = wrapper_mod.Wrapper(
        balls=ball_specs,
        arches=arch_specs,
        lighting=lighting,
        texture=texture,
        seed=seed,
        show_viewer=show_viewer,
    )

    unknown = [name for name in cameras if name not in wrapper.env.cameras]
    if unknown:
        raise ValueError(f"unknown camera(s): {', '.join(unknown)}; expected one of {ALL_CAMERAS}")

    log = runlog_mod.RunLog.create(seed)
    video_paths = {name: log.path / f"rollout_{name}.mp4" for name in cameras}
    for name, path in video_paths.items():
        wrapper.env.cameras[name].start_recording(str(path), fps=fps)

    llm = LLMSettings(
        client=OpenAI(api_key=main_mod.LLM_API_KEY, base_url=main_mod.LLM_BASE_URL),
        model_name=model,
        effort=effort,
        temperature=temperature,
        usage=usage_mod.UsageLog.create(log.path.name, model=model, effort=effort, temperature=temperature),
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
                "video": {name: path.name for name, path in video_paths.items()},
                "scene": {
                    "balls": [spec["name"] for spec in ball_specs],
                    "arches": [spec["name"] for spec in arch_specs],
                    "lighting": lighting,
                    "texture": texture,
                },
            },
            wrapper.env,
        )
        for attempt in range(max_attempts):
            log.set_attempt(attempt)
            try:
                apis = main_mod.build_api_doc(wrapper.namespace())
                code, skills = main_mod.generate_policy_code(
                    llm, task, settings, prompts_mod.SYSTEM_PROMPTS, prompts_mod.SKILLS, apis, attempt=attempt
                )
                log.note_skills(skills)
                log.save_policy(code)
                main_mod.run_policy(
                    llm,
                    code,
                    runlog_mod.logged_namespace(wrapper, log, timeout_s=max_seconds),
                    apis,
                    prompts_mod.SYSTEM_PROMPTS,
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
                    stop_reason = "timeout" if isinstance(exc, runlog_mod.AttemptTimeout) else "exception"
    except Exception:
        if not success:
            stop_reason = "exception"
        raise
    finally:
        for name in video_paths:
            try:
                wrapper.env.cameras[name].stop_recording()
            except Exception:
                pass
        log.finish(success, stop_reason, wrapper.env)
    try:
        review_mod.review_rollout(llm, log.path)
    except Exception as exc:
        print(f"[warn] review_rollout failed, skills not updated for this run: {exc}")
    print(f"success: {success} ({stop_reason})")
    for name, path in video_paths.items():
        print(f"  {name}: {path}")
    return log.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=sorted(PRESETS), help="Scene preset from rollout.py. Random if omitted.")
    parser.add_argument("--no-easymode", action="store_true", help="Film the real cap/ pipeline instead of cap_easymode.")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=DEFAULT_CAMERAS,
        choices=ALL_CAMERAS,
        help=f"Cameras to record, each to its own file. Default: {DEFAULT_CAMERAS}.",
    )
    parser.add_argument("--lighting", choices=CINEMATIC_LIGHTING, default=None, help="Random pick if omitted.")
    parser.add_argument(
        "--texture",
        choices=CINEMATIC_TEXTURES,
        default=None,
        help="Uses the pattern's own texture if omitted, falling back to leafy_glass.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Wall-clock budget per attempt. Uses the chosen module's default if omitted; 0 disables it.",
    )
    parser.add_argument("--viewer", action="store_true", help="Show the live viewer window while recording.")
    parser.add_argument("--seed", type=int, default=None, help="Scene seed. Random if omitted.")
    args = parser.parse_args()

    easymode = not args.no_easymode
    main_mod, _, _, _, _, _ = _select_backend(easymode)

    name = args.pattern or random.choice(list(PRESETS))
    preset = dict(PRESETS[name])
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    lighting = args.lighting or random.Random(seed).choice(CINEMATIC_LIGHTING)
    texture = args.texture or preset.get("texture", CINEMATIC_TEXTURES[0])
    print(f"pattern: {name}")
    print(f"mode: {'easymode' if easymode else 'cap'}")
    print(f"lighting: {lighting}")
    print(f"texture: {texture}")
    print(f"cameras: {', '.join(args.cameras)}")
    print(f"seed: {seed}")

    preset.pop("lighting", None)
    preset.pop("texture", None)
    path = film_rollout(
        **preset,
        lighting=lighting,
        texture=texture,
        cameras=args.cameras,
        fps=args.fps,
        easymode=easymode,
        model=args.model or main_mod.DEFAULT_MODEL_NAME,
        effort=args.effort or main_mod.DEFAULT_EFFORT,
        temperature=args.temperature if args.temperature is not None else main_mod.DEFAULT_TEMPRATURE,
        max_attempts=args.max_attempts or main_mod.MAX_ATTEMPTS,
        max_seconds=main_mod.DEFAULT_MAX_SECONDS if args.max_seconds is None else (args.max_seconds or None),
        show_viewer=args.viewer,
        seed=seed,
    )
    print(f"log: {path}")


if __name__ == "__main__":
    main()
