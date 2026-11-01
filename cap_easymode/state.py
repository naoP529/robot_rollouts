"""1ロールアウトの探索記憶。xy はメートル、yaw はラジアン。

easy mode: オブジェクトの位置は常に get_loc がシミュレータから直接読むので、
cap/state.py の Observed/Unseen (vision ベースの信念管理) は不要。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Pose:
    x: float
    y: float
    yaw: float | None = None

    def to_dict(self) -> dict:
        data = {"x": self.x, "y": self.y}
        if self.yaw is not None:
            data["yaw"] = self.yaw
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(float(data["x"]), float(data["y"]), _optional_float(data.get("yaw")))


@dataclass
class Robot:
    pose: Pose
    path: list[Pose] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pose": self.pose.to_dict(),
            "path": [pose.to_dict() for pose in self.path],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Robot":
        pose = Pose.from_dict(data["pose"])
        path = [Pose.from_dict(item) for item in data.get("path", [])]
        _require_yaw(pose, "robot.pose")
        for index, item in enumerate(path):
            _require_yaw(item, f"robot.path[{index}]")
        return cls(pose, path)


@dataclass
class RolloutState:
    robot: Robot

    def to_dict(self) -> dict:
        return {"robot": self.robot.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutState":
        return cls(Robot.from_dict(data["robot"]))


def save(state: RolloutState, path: str | Path) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), indent=2) + "\n")


def load(path: str | Path) -> RolloutState:
    return RolloutState.from_dict(json.loads(Path(path).read_text()))


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _require_yaw(pose: Pose, where: str) -> None:
    if pose.yaw is None:
        raise ValueError(f"{where} requires yaw")
