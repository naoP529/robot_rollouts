import torch
from torch import nn


class Command2Loc(nn.Module):
    """秒数と速度コマンド (duration, vx, vy, vr) から変位 (dx, dy, dr) を予測する。"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.mlp(x)


class Loc2Command(nn.Module):
    """秒数と目標変位 (duration, dx, dy, dr) から速度コマンド (vx, vy, vr) を予測する。"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.mlp(x)


def load_checkpoint(model, path, map_location="cpu"):
    """Colab が保存する best_checkpoint.pth / last_checkpoint.pth を読み、eval にする。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
