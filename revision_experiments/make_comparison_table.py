from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, TABLE_DIR, dataframe_to_markdown, ensure_output_dirs


DEFAULT_INPUTS = [
    TABLE_DIR / "baseline_eval_4mines.csv",
    PROJECT_ROOT / "dqn" / "csv" / "eval_dqn_4mines.csv",
    PROJECT_ROOT / "caedqn" / "csv" / "eval_caedqn_4mines.csv",
]


def load_eval_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ["logical_moves", "forced_guesses", "forced_guess_rate"]:
        if column not in df.columns:
            df[column] = pd.NA
    return df[
        [
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
    ]


def summarize(df: pd.DataFrame, by_seed: bool) -> pd.DataFrame:
    group_cols = ["algorithm", "eval_seed"] if by_seed else ["algorithm"]
    summary = (
        df.groupby(group_cols, as_index=False)
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
    return summary


def write_markdown_table(path: Path, overall: pd.DataFrame) -> None:
    table = overall.copy()
    preferred_order = {
        "Random Picker": 0,
        "Rule-Based Deterministic": 1,
        "DQN": 2,
        "CAE-DQN": 3,
    }
    table["order"] = table["algorithm"].map(preferred_order).fillna(99)
    table = table.sort_values("order").drop(columns=["order"])

    display = pd.DataFrame(
        {
            "Algorithm": table["algorithm"],
            "Episodes": table["episodes"],
            "Win Rate": (table["win_rate"] * 100).map(lambda x: f"{x:.1f}%"),
            "Avg Reward": table["avg_reward"].map(lambda x: f"{x:.3f}"),
            "Avg Clicks": table["avg_clicks"].map(lambda x: f"{x:.3f}"),
            "Avg Progress": table["avg_progress"].map(lambda x: f"{x:.3f}"),
            "Forced Guess Rate": table["forced_guess_rate"].map(
                lambda x: "-" if pd.isna(x) else f"{x * 100:.1f}%"
            ),
        }
    )

    lines = [
        "# Revision Experiment Comparison",
        "",
        "This table compares pure chance, deterministic logic, DQN, and CAE-DQN "
        "using the same 6x6 board, 4 mines, 5 evaluation seeds, and 500 episodes "
        "per seed protocol.",
        "",
        dataframe_to_markdown(display),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge baseline, DQN, and CAE-DQN evaluation results."
    )
    parser.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument(
        "--overall-output",
        type=Path,
        default=TABLE_DIR / "revision_comparison_overall.csv",
    )
    parser.add_argument(
        "--by-seed-output",
        type=Path,
        default=TABLE_DIR / "revision_comparison_by_seed.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=TABLE_DIR / "revision_comparison_table.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing input CSV(s). Run eval_baselines.py first if the baseline "
            f"CSV is missing:\n{missing_text}"
        )

    df = pd.concat([load_eval_csv(path) for path in args.inputs], ignore_index=True)
    by_seed = summarize(df, by_seed=True)
    overall = summarize(df, by_seed=False)

    by_seed.to_csv(args.by_seed_output, index=False)
    overall.to_csv(args.overall_output, index=False)
    write_markdown_table(args.markdown_output, overall)

    print(f"Saved by-seed comparison to {args.by_seed_output}")
    print(f"Saved overall comparison to {args.overall_output}")
    print(f"Saved Markdown table to {args.markdown_output}")


if __name__ == "__main__":
    main()
