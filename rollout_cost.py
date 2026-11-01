"""cap/usages/ と cap_easymode/usages/ に記録されたトークン数から、ロールアウト1回
あたりの平均コストを算出する。1 usageファイル = 1回のrollouts()呼び出し(= 1ロールアウト、
複数attemptやreview呼び出しも全部含む)。

料金は "gemini-4-argon"(2026-10-27時点、2026-11-30まで有効)。公開情報から:
    入力(非キャッシュ): $2.00 / 1Mトークン
    キャッシュ入力:     $0.10 / 1Mトークン
    出力(reasoning込み): $10.00 / 1Mトークン  ※thinking_level(低/中/高)で単価は変わらない
2026-12-01以降は$4.00 / $0.20 / $20.00に倍増する予定なので、その時期のログを計算する
場合はPRICINGを更新すること。

為替レートは2026-11-01時点のUSD/JPY ≈157.57(都度変わるのでUSD_JPYを更新すること)。

使い方:
    uv run python rollout_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# $ / 1M tokens
PRICING = {
    "gemini-4-argon": {
        "input": 2.00,
        "cached_input": 0.10,
        "output": 10.00,  # reasoning_tokensはcompletion_tokensの内訳であり別料金ではない
    },
}

USD_JPY = 157.57


def _cost_of_file(path: Path) -> float:
    data = json.loads(path.read_text())
    model = data.get("model")
    rates = PRICING.get(model)
    if rates is None:
        raise ValueError(f"{path}: unknown model {model!r}, add it to PRICING")
    totals = data.get("totals", {})
    prompt_tokens = totals.get("prompt_tokens", 0)
    cached_tokens = totals.get("cached_tokens", 0)
    completion_tokens = totals.get("completion_tokens", 0)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    return (
        uncached_tokens * rates["input"]
        + cached_tokens * rates["cached_input"]
        + completion_tokens * rates["output"]
    ) / 1_000_000


def report(label: str, usages_dir: Path) -> None:
    files = sorted(usages_dir.glob("*.json"))
    if not files:
        print(f"{label}: {usages_dir} に usage ファイルが無い")
        return
    costs = [_cost_of_file(f) for f in files]
    total = sum(costs)
    avg = total / len(costs)
    print(f"{label} ({usages_dir}):")
    print(f"  ロールアウト数: {len(costs)}")
    print(f"  合計: ${total:.4f} (¥{total * USD_JPY:,.1f})")
    print(f"  平均/ロールアウト: ${avg:.4f} (¥{avg * USD_JPY:,.1f})")
    print(f"  最小/最大: ${min(costs):.4f} / ${max(costs):.4f} (¥{min(costs) * USD_JPY:,.1f} / ¥{max(costs) * USD_JPY:,.1f})")


def main() -> None:
    report("cap", ROOT / "cap" / "usages")
    report("cap_easymode", ROOT / "cap_easymode" / "usages")


if __name__ == "__main__":
    main()
