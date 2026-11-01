"""1ロールアウトの探索記憶。xy はメートル、yaw はラジアン。"""

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
class Observed:
    id: str
    name: str
    query: str
    pose: Pose

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "query": self.query, "pose": self.pose.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "Observed":
        return cls(data["id"], data["name"], data["query"], Pose.from_dict(data["pose"]))


@dataclass
class Unseen:
    name: str
    query: str

    def to_dict(self) -> dict:
        return {"name": self.name, "query": self.query}

    @classmethod
    def from_dict(cls, data: dict) -> "Unseen":
        return cls(data["name"], data["query"])


@dataclass
class RolloutState:
    robot: Robot
    observed: list[Observed] = field(default_factory=list)
    unseen: list[Unseen] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "robot": self.robot.to_dict(),
            "observed": [item.to_dict() for item in self.observed],
            "unseen": [item.to_dict() for item in self.unseen],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RolloutState":
        return cls(
            Robot.from_dict(data["robot"]),
            [Observed.from_dict(item) for item in data.get("observed", [])],
            [Unseen.from_dict(item) for item in data.get("unseen", [])],
        )

    def find_observed(self, obj_id: str) -> Observed | None:
        return next((item for item in self.observed if item.id == obj_id), None)

    def query_of(self, name: str) -> str | None:
        for item in self.observed:
            if item.name == name:
                return item.query
        for item in self.unseen:
            if item.name == name:
                return item.query
        return None

    def next_instance_id(self, name: str) -> str:
        existing = {item.id for item in self.observed}
        n = 0
        while f"{name}_{n}" in existing:
            n += 1
        return f"{name}_{n}"

    def pop_unseen(self, name: str) -> Unseen | None:
        for index, item in enumerate(self.unseen):
            if item.name == name:
                return self.unseen.pop(index)
        return None


def strip_instance_suffix(id_or_name: str) -> str:
    """Undo next_instance_id: turn "name_N" back into "name" if it has a numeric suffix."""
    head, sep, tail = id_or_name.rpartition("_")
    if sep and tail.isdigit():
        return head
    return id_or_name


def is_instance_id(id_or_name: str) -> bool:
    """True if this looks like an instance id ("name_N") rather than a bare class name."""
    return strip_instance_suffix(id_or_name) != id_or_name


def require_instance_id(value: str, where: str) -> None:
    """Raise if `value` looks like a class name rather than an instance id from detect/search."""
    if not is_instance_id(value):
        raise ValueError(
            f"{where} expected an instance id (e.g. '{value}_0' from detect/search), "
            f"got the class name '{value}'."
        )


def require_class_name(value: str, where: str) -> None:
    """Raise if `value` looks like an instance id rather than a bare class name."""
    if is_instance_id(value):
        base = strip_instance_suffix(value)
        raise ValueError(f"{where} expected a class name (e.g. '{base}'), got the instance id '{value}'.")


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
