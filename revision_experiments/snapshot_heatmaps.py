from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BOARD_SIZE,
    FIGURE_DIR,
    N_MINES,
    SNAPSHOT_DIR,
    ensure_output_dirs,
    ensure_project_imports,
    mask_q_values_for_unknown,
    predict_q_values,
    quiet_tensorflow_logs,
)
from logic_tools import visible_board_from_state


def checkpoint_episode(path: Path) -> int:
    match = re.search(r"checkpoint_ep_(\d+)\.keras$", path.name)
    return int(match.group(1)) if match else -1


def prepare_fixed_state(seed: int, click_index: int):
    ensure_project_imports()
    from minesweeper_env import MinesweeperEnv

    env = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=seed)
    env.reset()
    env.step(click_index)
    return env.state_im.copy(), [tile.copy() for tile in env.state]


def plot_snapshots(checkpoints: list[Path], state_im: np.ndarray, state: list[dict], output: Path) -> None:
    import tensorflow as tf

    n = len(checkpoints)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    board = visible_board_from_state(state, BOARD_SIZE, BOARD_SIZE)

    for ax, checkpoint in zip(axes, checkpoints):
        model = tf.keras.models.load_model(checkpoint, compile=False)
        q_values = predict_q_values(model, state_im)
        masked = mask_q_values_for_unknown(state_im, q_values).reshape(BOARD_SIZE, BOARD_SIZE)
        im = ax.imshow(masked, cmap="viridis")
        ep = checkpoint_episode(checkpoint)
        ax.set_title(f"Episode {ep:,}")
        ax.set_xticks(range(BOARD_SIZE))
        ax.set_yticks(range(BOARD_SIZE))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                label = "?" if board[r, c] == "?" else str(board[r, c])
                if np.isnan(masked[r, c]):
                    ax.text(c, r, label, ha="center", va="center", color="black", fontsize=9)
                else:
                    ax.text(c, r, f"{masked[r, c]:.1f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(checkpoints):]:
        ax.axis("off")

    fig.suptitle("Q-value Evolution on a Fixed Board State", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Q-value heatmaps from saved checkpoints.")
    parser.add_argument("--algorithm", choices=["dqn", "caedqn"], default="caedqn")
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--board-seed", type=int, default=42)
    parser.add_argument("--click-index", type=int, default=21)
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quiet_tensorflow_logs()
    ensure_output_dirs()

    snapshot_dir = args.snapshot_dir or (SNAPSHOT_DIR / args.algorithm)
    output = args.output or (FIGURE_DIR / f"snapshot_heatmaps_{args.algorithm}.png")
    checkpoints = sorted(snapshot_dir.glob("checkpoint_ep_*.keras"), key=checkpoint_episode)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {snapshot_dir}. Run train_snapshots.py first."
        )

    if len(checkpoints) > args.max_checkpoints:
        indices = np.linspace(0, len(checkpoints) - 1, args.max_checkpoints).round().astype(int)
        checkpoints = [checkpoints[i] for i in indices]

    state_im, state = prepare_fixed_state(args.board_seed, args.click_index)
    plot_snapshots(checkpoints, state_im, state, output)
    print(f"Saved snapshot heatmap figure to {output}")


if __name__ == "__main__":
    main()
