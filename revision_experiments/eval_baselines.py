from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline_agents import RandomAgent, RuleBasedDeterministicAgent
from common import (
    BOARD_SIZE,
    EPISODES_PER_SEED,
    EVAL_SEEDS,
    N_MINES,
    TABLE_DIR,
    ensure_output_dirs,
    ensure_project_imports,
)


ensure_project_imports()
from minesweeper_env import MinesweeperEnv  # noqa: E402


def evaluate_agent(agent_cls, algorithm_name: str, episodes: int, seeds: list[int]):
    rows: list[dict] = []

    for eval_seed in seeds:
        env = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=eval_seed)
        agent = agent_cls(env, seed=10_000 + eval_seed) if agent_cls is RandomAgent else agent_cls(env)

        for episode in range(1, episodes + 1):
            env.reset()
            if hasattr(agent, "reset_episode"):
                agent.reset_episode()

            done = False
            total_reward = 0.0
            n_steps = 0

            while not done:
                action = int(agent.get_action(env.state_im))
                _, reward, done = env.step(action)
                total_reward += reward
                n_steps += 1

                if n_steps > env.n_tiles:
                    raise RuntimeError(
                        f"{algorithm_name} exceeded board action count on seed {eval_seed}, "
                        f"episode {episode}."
                    )

            logical_moves = int(getattr(agent, "logical_moves", 0))
            forced_guesses = int(getattr(agent, "forced_guesses", 0))
            rows.append(
                {
                    "algorithm": algorithm_name,
                    "board_size": BOARD_SIZE,
                    "n_mines": N_MINES,
                    "eval_seed": eval_seed,
                    "episode": episode,
                    "win": 1 if env.n_wins > 0 else 0,
                    "progress": env.n_progress,
                    "total_reward": round(total_reward, 2),
                    "n_clicks": env.n_clicks,
                    "logical_moves": logical_moves,
                    "forced_guesses": forced_guesses,
                    "forced_guess_rate": round(forced_guesses / max(1, env.n_clicks), 4),
                }
            )

        print(f"{algorithm_name}: completed seed {eval_seed} ({episodes} episodes)")

    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "algorithm",
        "board_size",
        "n_mines",
        "eval_seed",
        "episode",
        "win",
        "progress",
        "total_reward",
        "n_clicks",
        "logical_moves",
        "forced_guesses",
        "forced_guess_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["algorithm", "eval_seed"], as_index=False)
        .agg(
            episodes=("episode", "count"),
            win_rate=("win", "mean"),
            avg_reward=("total_reward", "mean"),
            avg_clicks=("n_clicks", "mean"),
            avg_progress=("progress", "mean"),
            avg_logical_moves=("logical_moves", "mean"),
            avg_forced_guesses=("forced_guesses", "mean"),
            forced_guess_rate=("forced_guess_rate", "mean"),
        )
        .round(4)
    )

    overall = (
        df.groupby(["algorithm"], as_index=False)
        .agg(
            eval_seed=("eval_seed", lambda s: "overall"),
            episodes=("episode", "count"),
            win_rate=("win", "mean"),
            avg_reward=("total_reward", "mean"),
            avg_clicks=("n_clicks", "mean"),
            avg_progress=("progress", "mean"),
            avg_logical_moves=("logical_moves", "mean"),
            avg_forced_guesses=("forced_guesses", "mean"),
            forced_guess_rate=("forced_guess_rate", "mean"),
        )
        .round(4)
    )

    pd.concat([summary, overall], ignore_index=True).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Random Picker and deterministic rule-based baselines."
    )
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLE_DIR / "baseline_eval_4mines.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=TABLE_DIR / "baseline_eval_4mines_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    all_rows: list[dict] = []
    all_rows.extend(evaluate_agent(RandomAgent, RandomAgent.name, args.episodes, args.seeds))
    all_rows.extend(
        evaluate_agent(
            RuleBasedDeterministicAgent,
            RuleBasedDeterministicAgent.name,
            args.episodes,
            args.seeds,
        )
    )

    write_rows(args.output, all_rows)
    write_summary(args.summary_output, all_rows)

    print(f"\nSaved per-episode baseline results to {args.output}")
    print(f"Saved baseline summary to {args.summary_output}")


if __name__ == "__main__":
    main()
