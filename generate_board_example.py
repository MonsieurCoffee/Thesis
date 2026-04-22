# Run this cell multiple times to generate new random Minesweeper boards
import random
import numpy as np
from minesweeper_env import MinesweeperEnv

def draw_full_board(env):
    rows, cols = env.n_rows, env.n_cols
    lines = ["   " + " ".join(f"{c:2}" for c in range(cols))]
    lines.append("  +" + "---+" * cols)
    for r in range(rows):
        row = [f"{r} |"]
        for c in range(cols):
            v = env.board[r, c]
            row.append(" B |" if v == 'B' else f" {v} |")
        lines.append("".join(row))
        lines.append("  +" + "---+" * cols)
    return "\n".join(lines)

def draw_visible_board(env):
    rows, cols = env.n_rows, env.n_cols
    lines = ["   " + " ".join(f"{c:2}" for c in range(cols))]
    lines.append("  +" + "---+" * cols)
    for r in range(rows):
        row = [f"{r} |"]
        for c in range(cols):
            v = env.state[r * cols + c]['value']
            if v == 'U':   row.append(" ? |")
            elif v == 'B': row.append(" B |")
            else:          row.append(f" {v} |")
        lines.append("".join(row))
        lines.append("  +" + "---+" * cols)
    return "\n".join(lines)

BOARD_SIZE = 6
N_MINES = 4

# Generate a random seed for this run
seed = random.randint(0, 2**31 - 1)
print(f"Using seed: {seed}\n")

# Create three environment instances with the SAME seed
env_hidden = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=seed)
env_click  = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=seed)
env_true   = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=seed)

# 1. Hidden board (all unknown)
print("=" * 50)
print("HIDDEN BOARD (all unknown)")
print("=" * 50)
print(draw_visible_board(env_hidden))

# 2. Board after first click (cell 0 is guaranteed safe)
env_click.step(0)
print("\n" + "=" * 50)
print("BOARD AFTER FIRST CLICK (clicked cell 0)")
print("=" * 50)
print(draw_visible_board(env_click))

# 3. True board (mines as 'B')
print("\n" + "=" * 50)
print("TRUE BOARD (mines are B)")
print("=" * 50)
print(draw_full_board(env_true))