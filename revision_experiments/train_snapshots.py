from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from common import (
    BOARD_SIZE,
    N_MINES,
    SNAPSHOT_DIR,
    TABLE_DIR,
    ensure_output_dirs,
    ensure_project_imports,
    quiet_tensorflow_logs,
)


def build_agent(algorithm: str, env, model_seed: int):
    if algorithm == "dqn":
        ensure_project_imports("dqn")
        from dqn_agent import DQNAgent

        return DQNAgent(env, model_name="revision_dqn_snapshot", seed=model_seed)

    if algorithm == "caedqn":
        ensure_project_imports("caedqn")
        from caedqn_agent import CAEDQNAgent

        return CAEDQNAgent(env, model_name="revision_caedqn_snapshot", seed=model_seed)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a short DQN/CAE-DQN run and save checkpoint snapshots."
    )
    parser.add_argument("--algorithm", choices=["dqn", "caedqn"], default="caedqn")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--env-seed", type=int, default=2004)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Defaults to revision_experiments/outputs/snapshots/<algorithm>.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Defaults to revision_experiments/outputs/tables/snapshot_training_<algorithm>.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quiet_tensorflow_logs()
    ensure_output_dirs()
    ensure_project_imports()

    import tensorflow as tf
    from minesweeper_env import MinesweeperEnv

    np.random.seed(args.model_seed)
    tf.random.set_seed(args.model_seed)

    snapshot_dir = args.snapshot_dir or (SNAPSHOT_DIR / args.algorithm)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    log_output = args.log_output or (TABLE_DIR / f"snapshot_training_{args.algorithm}.csv")

    env = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=args.env_seed)
    agent = build_agent(args.algorithm, env, args.model_seed)

    initial_path = snapshot_dir / "checkpoint_ep_000000.keras"
    agent.model.save(initial_path)
    print(f"Saved initial checkpoint to {initial_path}")

    start_time = time.time()
    recent_wins: list[int] = []
    recent_progress: list[int] = []
    recent_rewards: list[float] = []

    with log_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "total_reward",
                "progress",
                "win",
                "n_clicks",
                "recent_win_rate",
                "recent_median_progress",
                "recent_median_reward",
                "time_elapsed",
                "checkpoint_path",
            ]
        )

        for episode in range(1, args.episodes + 1):
            env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                current_state = env.state_im
                action = agent.get_action(current_state)
                new_state, reward, done = env.step(action)

                if args.algorithm == "caedqn":
                    bonus = agent.compute_bonus(new_state) if not done else 0.0
                    stored_reward = reward + bonus
                else:
                    stored_reward = reward

                episode_reward += stored_reward
                agent.update_replay_memory(
                    (current_state, action, stored_reward, new_state, done)
                )
                agent.train(done)

            win = 1 if env.n_wins > 0 else 0
            recent_wins.append(win)
            recent_progress.append(env.n_progress)
            recent_rewards.append(episode_reward)

            checkpoint_path = ""
            if episode % args.save_every == 0:
                path = snapshot_dir / f"checkpoint_ep_{episode:06d}.keras"
                agent.model.save(path)
                checkpoint_path = str(path)
                window = min(args.save_every, len(recent_wins))
                recent_win_rate = float(np.mean(recent_wins[-window:]))
                recent_median_progress = float(np.median(recent_progress[-window:]))
                recent_median_reward = float(np.median(recent_rewards[-window:]))
                elapsed = time.time() - start_time

                print(
                    f"{args.algorithm.upper()} episode {episode}/{args.episodes} | "
                    f"win_rate={recent_win_rate:.3f} | "
                    f"median_progress={recent_median_progress:.1f} | "
                    f"time={elapsed:.1f}s"
                )

                writer.writerow(
                    [
                        episode,
                        round(episode_reward, 4),
                        env.n_progress,
                        win,
                        env.n_clicks,
                        round(recent_win_rate, 4),
                        round(recent_median_progress, 4),
                        round(recent_median_reward, 4),
                        round(elapsed, 2),
                        checkpoint_path,
                    ]
                )
                handle.flush()

    print(f"\nSnapshot training log saved to {log_output}")
    print(f"Checkpoints saved under {snapshot_dir}")


if __name__ == "__main__":
    main()
