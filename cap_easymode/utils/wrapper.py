from pathlib import Path

import genesis as gs
import torch

from cap_easymode.blocks.drive import pose_from_env
from cap_easymode.blocks.get_loc import get_loc
from cap_easymode.blocks.goto import goto
from cap_easymode.blocks.is_finished import is_finished
from cap_easymode.blocks.look_at import look_at
from cap_easymode.blocks.nudge import nudge
from cap_easymode.state import Robot, RolloutState
from envs.arcade import ArcadeEnv
from envs.cfg.lights import LIGHTING_PRESETS
from envs.cfg.model_data import ARCHES, BALLS
from models.loc import Loc2Command, load_checkpoint
from models.walker import SimpleWalker

ROOT = Path(__file__).resolve().parents[2]
WALKER_PATH = ROOT / "weights" / "thefirststep" / "policy.pth"
LOC2COMMAND_PATH = ROOT / "weights" / "loc2command" / "best_checkpoint.pth"
TEXTURE_ROOT = ROOT / "assets" / "texture"
DEFAULT_TEXTURE = "leafy_glass"

ENV_CFG = {
    "num_actions": 12,
    "default_joint_angles": {
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
    "kp": 20.0,
    "kd": 0.5,
    "termination_if_roll_greater_than": 30,
    "termination_if_pitch_greater_than": 30,
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 120.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": True,
    "clip_actions": 100.0,
}

OBS_CFG = {
    "obs_scales": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    },
}


def _lighting(name: str = "clear_noon"):
    for preset in LIGHTING_PRESETS:
        if preset["name"] == name:
            return preset
    return LIGHTING_PRESETS[0]


def _texture_dir(texture: str | Path | None) -> Path:
    if texture is None:
        return TEXTURE_ROOT / DEFAULT_TEXTURE
    path = Path(texture)
    if path.is_dir():
        return path
    return TEXTURE_ROOT / path.name


def _init_genesis():
    if getattr(gs, "_initialized", False):
        return
    try:
        gs.init(backend=gs.gpu)
    except Exception:
        gs.init(backend=gs.cpu)


class Wrapper:
    """Load env and weights. Bind blocks so the LMP never sees them.

    Easy mode: object and robot positions come straight from the simulator, not from
    OWLv2 vision or a learned command2loc localizer, so there is no detect/search and
    no belief state to keep in sync with the physics.
    """

    def __init__(
        self,
        balls=None,
        arches=None,
        lighting: str = "clear_noon",
        texture: str | Path | None = DEFAULT_TEXTURE,
        seed: int = 0,
        show_viewer: bool = True,
        device=None,
    ):
        _init_genesis()
        self.device = torch.device(device or gs.device)
        self.env = ArcadeEnv(
            num_envs=1,
            env_cfg=ENV_CFG,
            obs_cfg=OBS_CFG,
            balls=list(balls if balls is not None else BALLS),
            arches=list(arches if arches is not None else ARCHES),
            lighting=_lighting(lighting),
            texture_dir=_texture_dir(texture),
            show_viewer=show_viewer,
            rendered_envs_idx=(0,) if show_viewer else None,
            follow_robot=False,
            seed=seed,
        )
        self.env.reset()
        self.walker = self._load_walker()
        self.loc2command = load_checkpoint(Loc2Command(), LOC2COMMAND_PATH, self.device).to(self.device)
        self.state = self._init_state()

    def _load_walker(self) -> SimpleWalker:
        walker = SimpleWalker(45, 12)
        walker.load_state_dict(torch.load(WALKER_PATH, weights_only=True, map_location=self.device))
        walker.to(self.device)
        walker.eval()
        return walker

    def _init_state(self) -> RolloutState:
        return RolloutState(Robot(pose_from_env(self.env)))

    def get_loc(self, name: str):
        """Return the exact pose of the robot (name="robot") or a named ball/arch,
        read directly from the simulator."""
        return get_loc(self.env, self.state, name)

    def look_at(self, yaw: float):
        """Turn in place to a yaw and update the robot pose in state."""
        return look_at(self.env, self.walker, self.state, yaw)

    def goto(self, x: float, y: float, yaw: float | None = None):
        """Walk to a world pose and update the robot pose in state."""
        return goto(self.env, self.walker, self.loc2command, self.state, x, y, yaw)

    def nudge(self, name: str, xy: tuple[float, float], speed: float = 1.0, duration_s: float = 0.5):
        """Push the named ball toward xy by walking into it, then return its exact
        post-push pose read straight from the simulator."""
        return nudge(self.env, self.walker, self.state, name, xy, speed, duration_s)

    def is_finished(self, ball: str, arch: str):
        """Return whether the ball has passed through the arch."""
        return is_finished(self.env, self.state, ball, arch)

    def namespace(self) -> dict:
        return {
            "get_loc": self.get_loc,
            "look_at": self.look_at,
            "goto": self.goto,
            "nudge": self.nudge,
            "is_finished": self.is_finished,
        }
