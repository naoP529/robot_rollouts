"""Run cap.main.rollouts() (or cap_easymode.main.rollouts() with --easymode) with a preset
scene, picked at random unless --pattern is given.

cap/main.py needs ball/arch/lighting/... specified every time, which gets tedious.
This script keeps a handful of ready-made scenes and launches one of them.
"""

import argparse
import random

PRESETS = {
    "red_through_blue": dict(
        target_ball="ball_red", target_arch="arch_blue", lighting="clear_noon", texture="clean_asphalt"
    ),
    "yellow_through_green_morning": dict(
        target_ball="ball_yellow", target_arch="arch_green", lighting="morning", texture="leafy_glass"
    ),
    "cyan_through_purple_sunset": dict(
        target_ball="ball_cyan", target_arch="arch_purple", lighting="sunset", texture="brick_pavement"
    ),
    "black_through_white_moonlight": dict(
        target_ball="ball_black", target_arch="arch_white", lighting="moonlight", texture="gravel_floor"
    ),
    "pink_through_orange_overcast": dict(
        target_ball="ball_pink", target_arch="arch_orange", lighting="overcast", texture="clean_asphalt"
    ),
    "green_through_cyan_golden_hour": dict(
        target_ball="ball_green", target_arch="arch_cyan", lighting="golden_hour", texture="walnut_veneer"
    ),
    "white_through_black_snowfield": dict(
        target_ball="ball_white", target_arch="arch_black", lighting="snowfield", texture="snow"
    ),
    "simple_pair_desert": dict(
        target_ball="ball_blue",
        target_arch="arch_red",
        balls=["ball_red", "ball_blue"],
        arches=["arch_red", "arch_blue"],
        lighting="desert_noon",
        texture="gravel_floor",
    ),
}


def main() -> None:
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument(
        "--easymode",
        action="store_true",
        help="Use cap_easymode (exact get_loc from the simulator, no owl/command2loc) instead of cap.",
    )
    mode_args, _ = mode_parser.parse_known_args()
    if mode_args.easymode:
        from cap_easymode.main import DEFAULT_EFFORT, DEFAULT_MODEL_NAME, DEFAULT_TEMPRATURE, rollouts
    else:
        from cap.main import DEFAULT_EFFORT, DEFAULT_MODEL_NAME, DEFAULT_TEMPRATURE, rollouts

    parser = argparse.ArgumentParser(description=__doc__, parents=[mode_parser])
    parser.add_argument("--pattern", choices=sorted(PRESETS), help="Run this preset instead of a random one.")
    parser.add_argument("--list", action="store_true", help="List presets and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPRATURE)
    parser.add_argument("--seed", type=int, default=None, help="Scene seed. Random if omitted.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--max-attempts", type=int, default=None, help="Cap retries. Uses the chosen module's default if omitted."
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Wall-clock budget per attempt. Uses the chosen module's default if omitted; 0 disables it.",
    )
    args = parser.parse_args()

    if args.list:
        for name in sorted(PRESETS):
            print(f"{name}: {PRESETS[name]}")
        return

    name = args.pattern or random.choice(list(PRESETS))
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"pattern: {name}")
    print(f"mode: {'easymode' if args.easymode else 'cap'}")
    print(f"seed: {seed}")

    kwargs = dict(
        model=args.model,
        effort=args.effort,
        temperature=args.temperature,
        seed=seed,
        show_viewer=not args.no_viewer,
        record_video=not args.no_video,
    )
    if args.max_attempts is not None:
        kwargs["max_attempts"] = args.max_attempts
    if args.max_seconds is not None:
        kwargs["max_seconds"] = args.max_seconds or None
    path = rollouts(**kwargs, **PRESETS[name])
    print(f"log: {path}")


if __name__ == "__main__":
    main()
