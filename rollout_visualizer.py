import genesis as gs
from envs.go2env import Go2Env
from models.walker import SimpleWalker
import argparse
import torch
import os, json
from pathlib import Path

weights_dir = Path("weights")

env_config = {
    "num_actions": 12,
    # joint/link names
    "default_joint_angles": {  # [rad]
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    "joint_names": [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    # PD
    "kp": 20.0,
    "kd": 0.5,
    # termination
    "termination_if_roll_greater_than": 30,  # degree
    "termination_if_pitch_greater_than": 30,
    # base pose
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 20.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": True,
    "clip_actions": 100.0,
}

obs_config={
    "obs_scales": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_name", type=str)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--single-command",
        type=str,
        default='{"vel_front":0.1, "vel_right":0.0, "vel_angular":0.1}',
        help="json format",
    )
    parser.add_argument("--cmd-file", action="store_true")
    parser.add_argument("--cmd-path", type=str, default="command.txt")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    train_name = args.train_name
    train_dir = weights_dir / train_name

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(train_dir)
    
    gs.init(backend=gs.gpu)

    model = SimpleWalker(45, 12)
    weight_path = train_dir / "policy.pth"
    state = torch.load(weight_path, weights_only=True, map_location=gs.device)
    model.load_state_dict(state)
    model.to(gs.device)
    model.eval()

    env = Go2Env(num_envs=args.num_envs, env_cfg=env_config, obs_cfg=obs_config, show_viewer=args.viewer)

    cmd_list = []
    if(not args.cmd_file):
        cmd_single = json.loads(args.single_command)
        cmd_list.append(
            {
                "duration":4.0,
                "vel_front":cmd_single["vel_front"],
                "vel_right":cmd_single["vel_right"],
                "vel_angular":cmd_single["vel_angular"]
            }
        )
    else:
        with open(args.cmd_path, encoding="utf-8") as f:
            content = f.read()
            parsed = [r.strip() for r in content.split("\n")]
            for i in range(2, len(parsed)-1):
                row = parsed[i]
                if row == "" or row == None:
                    continue
                values = [float(v) for v in row.split("|")]
                if len(values) != 4:
                    raise ValueError(f"invalid commands. {i+1} row: {row}")
                cmd_list.append(
                    {
                        "duration":values[0],
                        "vel_front":values[1],
                        "vel_right":values[2],
                        "vel_angular":values[3]
                    }
                )

    obs = env.reset()
    for cmd in cmd_list:
        env.set_commands([cmd["vel_front"], cmd["vel_right"], cmd["vel_angular"]])
        obs = env.get_observations()
        n_steps = int(round(cmd["duration"] / env.dt))
        for _ in range(n_steps):
            with torch.no_grad():
                action = model(obs).mean
            obs, done = env.step(action)
            if done.any():
                env.reset(done)
                obs = env.get_observations()


if __name__ == "__main__":
    main()
        