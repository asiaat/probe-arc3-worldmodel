#!/usr/bin/env python3
"""Fetch real human baselines from the ARC-AGI-3 SDK scorecard.

This replaces the fake hardcoded HUMAN_BASELINES in the old repo with
actual values from the competition. Saves to data/human_baselines.json.

The SDK exposes human baselines differently across versions. This script
tries several access paths and falls back to reasonable defaults with a
warning so development can continue.

Usage:
    python scripts/fetch_baselines.py
    python scripts/fetch_baselines.py --games ls20,ft09
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import arc_agi
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Fallback values — used if SDK can't provide real baselines. These are
# placeholders; do NOT use these numbers to report scores.
FALLBACK_BASELINES = {
    "ls20": [8, 12, 15, 20, 25],
    "ft09": [6, 10, 14, 18, 22],
    "default": [10, 15, 20, 25, 30],
}


def try_fetch_baseline_for_game(arc, game_id: str) -> list[int] | None:
    """Attempt to fetch the human baseline for a single game.

    Returns:
        List of second-best human action counts per level, or None if
        unavailable in this SDK version.
    """
    # Try a series of plausible SDK access patterns
    try:
        env = arc.make(game_id)
    except Exception as e:
        print(f"[WARN] Could not make game {game_id}: {e}")
        return None

    # Pattern 1: env.human_baseline
    for attr in ("human_baseline", "baseline", "human_actions", "scorecard"):
        if hasattr(env, attr):
            val = getattr(env, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            if isinstance(val, list | tuple) and all(isinstance(x, int | float) for x in val):
                return [int(x) for x in val]
            if isinstance(val, dict):
                # Might be {"level_1": N, "level_2": N, ...}
                try:
                    sorted_items = sorted(val.items())
                    return [int(v) for _, v in sorted_items]
                except Exception:
                    continue

    # Pattern 2: arc.get_baseline(game_id)
    for method_name in ("get_baseline", "get_human_baseline", "baseline_for"):
        if hasattr(arc, method_name):
            try:
                result = getattr(arc, method_name)(game_id)
                if isinstance(result, list | tuple):
                    return [int(x) for x in result]
            except Exception:
                continue

    # Pattern 3: arc.scorecard[game_id]
    for attr in ("scorecard", "baselines", "human_baselines"):
        if hasattr(arc, attr):
            sc = getattr(arc, attr)
            if callable(sc):
                try:
                    sc = sc()
                except Exception:
                    continue
            if isinstance(sc, dict) and game_id in sc:
                val = sc[game_id]
                if isinstance(val, list | tuple):
                    return [int(x) for x in val]

    return None


def fetch_all_baselines(games: list[str], output_path: Path) -> dict:
    """Fetch baselines for all games, persist to JSON, return the dict."""
    result = {
        "games": {},
        "source": "sdk" if HAS_SDK else "fallback",
        "notes": [],
    }

    if not HAS_SDK:
        print("[WARN] arc-agi SDK not installed. Using fallback placeholders.")
        print("[WARN] These are NOT real human baselines. Install the SDK and re-run.")
        for g in games:
            result["games"][g] = {
                "levels": FALLBACK_BASELINES.get(g, FALLBACK_BASELINES["default"]),
                "source": "fallback",
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        return result

    # SDK is available — try to get real baselines
    arc = arc_agi.Arcade()
    print(f"[INFO] SDK found. Attempting to fetch real baselines for {len(games)} games.")
    print()

    for g in games:
        print(f"[INFO] Fetching baseline for {g}...")
        baseline = try_fetch_baseline_for_game(arc, g)
        if baseline:
            print(f"[OK]   {g}: {baseline}")
            result["games"][g] = {
                "levels": baseline,
                "source": "sdk",
            }
        else:
            fallback = FALLBACK_BASELINES.get(g, FALLBACK_BASELINES["default"])
            print(f"[WARN] {g}: SDK did not provide baseline. Using fallback: {fallback}")
            print(f"[WARN] If you know the real baseline for {g}, edit "
                  f"{output_path} manually.")
            result["games"][g] = {
                "levels": fallback,
                "source": "fallback",
            }
            result["notes"].append(
                f"{g}: SDK did not provide baseline; using placeholder. "
                f"Replace with real scorecard data before relying on RHAE."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print()
    print(f"[DONE] Baselines saved to {output_path}")
    if result["notes"]:
        print("[WARN] Some baselines are fallback placeholders. See notes in the file.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch ARC-AGI-3 human baselines")
    parser.add_argument("--games", type=str, default="ls20,ft09",
                        help="Comma-separated game IDs")
    parser.add_argument("--output", type=str, default="./data/human_baselines.json")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    output = Path(args.output)

    result = fetch_all_baselines(games, output)

    print()
    print("Summary:")
    for g, info in result["games"].items():
        print(f"  {g}: {info['levels']}  ({info['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
