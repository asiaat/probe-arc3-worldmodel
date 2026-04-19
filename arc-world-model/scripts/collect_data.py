#!/usr/bin/env python3
"""Collect random-policy transitions into a replay buffer.

Usage:
    python scripts/collect_data.py --games ls20,ft09 --target 10000

This runs random-action episodes on the specified games until the target
transition count is reached. Transitions are saved incrementally to a
disk-backed ReplayBuffer — safe to Ctrl+C mid-run and resume.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from arcwm.env.arc_wrapper import ArcEnv, Transition, NUM_ACTIONS, HAS_SDK
from arcwm.env.replay_buffer import ReplayBuffer


def collect(
    games: list[str],
    target: int,
    max_steps: int,
    output_dir: str,
    seed: int = 42,
    resume: bool = True,
) -> None:
    """Collect transitions using a uniform random policy.

    Args:
        games: list of game IDs to rotate through
        target: target total transition count
        max_steps: max steps per episode (hard timeout)
        output_dir: directory for the replay buffer
        seed: RNG seed
        resume: if True and a buffer exists at output_dir, append to it
    """
    if not HAS_SDK:
        print("[WARN] arc-agi SDK not installed. Running with stub environment —")
        print("[WARN] collected transitions will not be meaningful.")

    out = Path(output_dir)
    rng = np.random.default_rng(seed)

    # Create or load buffer
    if out.exists() and (out / "meta.json").exists():
        if resume:
            print(f"[INFO] Resuming existing buffer at {out}")
            buf = ReplayBuffer.load(out)
        else:
            print(f"[ERROR] Buffer exists at {out}. Delete or pass --no-resume to overwrite.")
            sys.exit(1)
    else:
        # Pre-allocate with headroom
        capacity = max(target * 2, 100_000)
        print(f"[INFO] Creating new buffer at {out} (capacity={capacity:,})")
        buf = ReplayBuffer.create(out, capacity=capacity)

    initial_size = len(buf)
    print(f"[INFO] Starting size: {initial_size:,}  target: {target:,}")
    print(f"[INFO] Games: {games}")
    print(f"[INFO] Max steps per episode: {max_steps}")
    print()

    # Set up graceful shutdown — flush buffer on Ctrl+C
    stopping = {"flag": False}

    def _sigint_handler(signum, frame):
        print("\n[INFO] Interrupt received. Flushing buffer and exiting...")
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # Main collection loop
    start_time = time.time()
    episode_idx = 0
    last_flush = time.time()
    FLUSH_EVERY_SECONDS = 30

    while len(buf) < target and not stopping["flag"]:
        game_id = games[episode_idx % len(games)]
        episode_idx += 1

        try:
            env = ArcEnv(game_id)
            frame = env.reset()
        except Exception as e:
            print(f"[WARN] Failed to init game {game_id}: {e}")
            continue

        ep_steps = 0
        ep_reward = 0.0
        ep_wins = 0
        ep_terminals = 0

        for step in range(max_steps):
            if stopping["flag"]:
                break

            action = int(rng.integers(0, NUM_ACTIONS))
            next_frame, reward, terminal, info = env.step(action)

            t = Transition(
                frame=frame,
                action=action,
                reward=float(reward),
                next_frame=next_frame,
                terminal=bool(terminal),
                win=bool(info["win"]),
                game_id=game_id,
                step_idx=step,
            )

            try:
                buf.add(t)
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                break

            ep_steps += 1
            ep_reward += reward
            if info["win"]:
                ep_wins += 1
            if terminal:
                ep_terminals += 1

            frame = next_frame

            if terminal:
                break

            if len(buf) >= target:
                break

        env.close()

        # Periodic progress report and flush
        elapsed = time.time() - start_time
        rate = (len(buf) - initial_size) / max(elapsed, 1e-6)
        remaining = (target - len(buf)) / max(rate, 1e-6)

        print(
            f"[EP {episode_idx:4d}] game={game_id:6s}  "
            f"steps={ep_steps:3d}  reward={ep_reward:+.2f}  "
            f"wins={ep_wins}  terminal={ep_terminals}  "
            f"buffer={len(buf):,}/{target:,}  "
            f"rate={rate:.1f}/s  eta={remaining:.0f}s"
        )

        # Flush periodically so crashes don't lose much
        if time.time() - last_flush > FLUSH_EVERY_SECONDS:
            buf.flush()
            last_flush = time.time()

    # Final flush
    buf.flush()

    elapsed = time.time() - start_time
    print()
    print(f"[DONE] Collected {len(buf) - initial_size:,} new transitions in {elapsed:.0f}s")
    print(f"[DONE] Total buffer size: {len(buf):,}")
    print(f"[DONE] Buffer saved to: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect random-policy transitions")
    parser.add_argument("--games", type=str, default="ls20,ft09",
                        help="Comma-separated game IDs")
    parser.add_argument("--target", type=int, default=10_000,
                        help="Target transition count")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default="./data/replay",
                        help="Replay buffer directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true",
                        help="Fail if buffer already exists (default: resume)")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    if not games:
        print("[ERROR] No games specified.")
        return 1

    collect(
        games=games,
        target=args.target,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
