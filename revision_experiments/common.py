from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
SNAPSHOT_DIR = OUTPUT_ROOT / "snapshots"

BOARD_SIZE = 6
N_MINES = 4
EVAL_SEEDS = [1, 2, 3, 4, 5]
EPISODES_PER_SEED = 500


def ensure_project_imports(*extra_dirs: str) -> None:
    """Make project modules importable when scripts are run from any cwd."""
    paths = [PROJECT_ROOT, *(PROJECT_ROOT / d for d in extra_dirs)]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def ensure_output_dirs() -> None:
    for path in (OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, SNAPSHOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def quiet_tensorflow_logs() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


def unknown_indices_from_state_list(state: list[dict], n_cols: int) -> list[int]:
    return [
        idx
        for idx, tile in enumerate(state)
        if tile.get("value") == "U"
    ]


def unknown_indices_from_image(state_im: np.ndarray) -> list[int]:
    flat_unknown = state_im[:, :, 9].reshape(-1)
    return [int(i) for i, value in enumerate(flat_unknown) if value == 1.0]


def mask_q_values_for_unknown(state_im: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    masked = np.array(q_values, dtype=np.float64, copy=True).reshape(-1)
    unknown = np.zeros(masked.shape, dtype=bool)
    unknown[unknown_indices_from_image(state_im)] = True
    masked[~unknown] = np.nan
    return masked


def action_to_coord(action: int, n_cols: int = BOARD_SIZE) -> tuple[int, int]:
    return int(action // n_cols), int(action % n_cols)


def coord_to_action(row: int, col: int, n_cols: int = BOARD_SIZE) -> int:
    return int(row * n_cols + col)


def format_action(action: int, n_cols: int = BOARD_SIZE) -> str:
    row, col = action_to_coord(action, n_cols)
    return f"({row}, {col})"


def predict_q_values(model, state_im: np.ndarray) -> np.ndarray:
    """Return flat Q-values for both DQN and CAE-DQN Keras models."""
    state_batch = np.expand_dims(state_im, axis=0)
    outputs = model.predict(state_batch, verbose=0)
    if isinstance(outputs, (list, tuple)):
        q_values = outputs[0][0]
    else:
        q_values = outputs[0]
    return np.array(q_values, dtype=np.float64).reshape(-1)


def dataframe_to_markdown(df) -> str:
    """Small markdown table writer that does not require the tabulate package."""
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append([_markdown_cell(row[col]) for col in columns])
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _markdown_cell(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    text = str(value)
    return text.replace("|", "\\|")
