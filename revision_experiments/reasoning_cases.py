from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baseline_agents import RuleBasedDeterministicAgent
from common import (
    BOARD_SIZE,
    FIGURE_DIR,
    N_MINES,
    PROJECT_ROOT,
    TABLE_DIR,
    action_to_coord,
    ensure_output_dirs,
    ensure_project_imports,
    format_action,
    mask_q_values_for_unknown,
    predict_q_values,
    quiet_tensorflow_logs,
)
from logic_tools import ConstraintAnalysis, exact_frontier_probabilities, visible_board_from_state


plt = None
Rectangle = None


@dataclass
class CaseSnapshot:
    case_type: str
    seed: int
    step: int
    state_im: np.ndarray
    state: list[dict]
    true_board: np.ndarray
    analysis: ConstraintAnalysis


def clone_case(case_type: str, seed: int, step: int, env, analysis: ConstraintAnalysis) -> CaseSnapshot:
    return CaseSnapshot(
        case_type=case_type,
        seed=seed,
        step=step,
        state_im=env.state_im.copy(),
        state=copy.deepcopy(env.state),
        true_board=env.board.copy(),
        analysis=analysis,
    )


def search_cases(
    max_seed: int,
    max_steps: int,
    max_frontier: int,
    models: dict[str, object] | None = None,
    preferred_model: str = "CAE-DQN",
) -> tuple[CaseSnapshot, CaseSnapshot]:
    ensure_project_imports()
    from minesweeper_env import MinesweeperEnv

    reasoning_case: CaseSnapshot | None = None
    probability_case: CaseSnapshot | None = None

    for seed in range(1, max_seed + 1):
        env = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=seed)
        agent = RuleBasedDeterministicAgent(env)
        env.reset()
        agent.reset_episode()

        for step in range(max_steps + 1):
            analysis = exact_frontier_probabilities(
                env.state,
                BOARD_SIZE,
                BOARD_SIZE,
                N_MINES,
                known_mines=agent.known_mines,
                max_frontier=max_frontier,
            )

            has_revealed = any(tile["value"] != "U" for tile in env.state)
            if (
                has_revealed
                and reasoning_case is None
                and analysis.safe_cells
                and _reasoning_case_matches_model(
                    analysis, env.state_im, models, preferred_model
                )
            ):
                reasoning_case = clone_case("reasoning", seed, step, env, analysis)

            if has_revealed and probability_case is None:
                probs = [
                    prob
                    for cell, prob in analysis.mine_probabilities.items()
                    if cell not in analysis.mine_cells and cell not in analysis.safe_cells
                ]
                if (
                    analysis.used_exact_solver
                    and not analysis.safe_cells
                    and len(probs) >= 2
                    and (max(probs) - min(probs)) >= 0.05
                    and _probability_case_matches_model(
                        analysis, env.state_im, models, preferred_model
                    )
                ):
                    probability_case = clone_case("probability", seed, step, env, analysis)

            if reasoning_case is not None and probability_case is not None:
                return reasoning_case, probability_case

            if step >= max_steps:
                break
            action = agent.get_action(env.state_im)
            _, _, done = env.step(action)
            if done:
                break

    missing = []
    if reasoning_case is None:
        missing.append("reasoning")
    if probability_case is None:
        missing.append("probability")
    raise RuntimeError(f"Could not find case(s): {', '.join(missing)}")


def _best_model_coord(model, state_im: np.ndarray) -> tuple[int, int]:
    q_values = predict_q_values(model, state_im)
    masked = mask_q_values_for_unknown(state_im, q_values)
    action = int(np.nanargmax(masked))
    return action_to_coord(action, BOARD_SIZE)


def _preferred_model(models: dict[str, object] | None, preferred_model: str):
    if not models:
        return None
    return models.get(preferred_model) or next(iter(models.values()))


def _reasoning_case_matches_model(
    analysis: ConstraintAnalysis,
    state_im: np.ndarray,
    models: dict[str, object] | None,
    preferred_model: str,
) -> bool:
    model = _preferred_model(models, preferred_model)
    if model is None:
        return True
    return _best_model_coord(model, state_im) in analysis.safe_cells


def _probability_case_matches_model(
    analysis: ConstraintAnalysis,
    state_im: np.ndarray,
    models: dict[str, object] | None,
    preferred_model: str,
) -> bool:
    model = _preferred_model(models, preferred_model)
    if model is None:
        return True
    coord = _best_model_coord(model, state_im)
    selected_prob = analysis.mine_probabilities.get(coord)
    if selected_prob is None:
        return False
    min_prob = min(analysis.mine_probabilities.values())
    return selected_prob <= min_prob + 1e-9


def load_models(dqn_path: Path, caedqn_path: Path):
    quiet_tensorflow_logs()
    import tensorflow as tf

    return {
        "DQN": tf.keras.models.load_model(dqn_path, compile=False),
        "CAE-DQN": tf.keras.models.load_model(caedqn_path, compile=False),
    }


def plot_case(case: CaseSnapshot, models: dict[str, object], output_path: Path) -> list[dict]:
    _ensure_matplotlib()
    board = visible_board_from_state(case.state, BOARD_SIZE, BOARD_SIZE)
    prob_grid = np.full((BOARD_SIZE, BOARD_SIZE), np.nan, dtype=float)
    for (row, col), probability in case.analysis.mine_probabilities.items():
        prob_grid[row, col] = probability

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    _plot_board(axes[0], board, case)

    action_rows: list[dict] = []
    for ax, (name, model) in zip(axes[1:3], models.items()):
        q_values = predict_q_values(model, case.state_im)
        masked = mask_q_values_for_unknown(case.state_im, q_values)
        best_action = int(np.nanargmax(masked))
        best_row, best_col = action_to_coord(best_action, BOARD_SIZE)
        _plot_q_heatmap(ax, masked.reshape(BOARD_SIZE, BOARD_SIZE), board, case, name, best_action)
        action_rows.append(
            {
                "case_type": case.case_type,
                "seed": case.seed,
                "step": case.step,
                "model": name,
                "selected_action": best_action,
                "selected_coord": format_action(best_action, BOARD_SIZE),
                "q_value": round(float(q_values[best_action]), 6),
                "mine_probability": round(
                    float(case.analysis.mine_probabilities.get((best_row, best_col), np.nan)),
                    6,
                ),
                "is_guaranteed_safe": (best_row, best_col) in case.analysis.safe_cells,
                "is_guaranteed_mine": (best_row, best_col) in case.analysis.mine_cells,
                "true_value": str(case.true_board[best_row, best_col]),
                "used_exact_solver": case.analysis.used_exact_solver,
                "frontier_size": case.analysis.frontier_size,
                "valid_assignments": case.analysis.valid_assignments,
            }
        )

    _plot_probability_heatmap(axes[3], prob_grid, board, case)

    title = (
        "Reasoning Case: guaranteed-safe action exists"
        if case.case_type == "reasoning"
        else "Probability Case: no guaranteed-safe action; choose lower-risk cell"
    )
    fig.suptitle(f"{title} (seed={case.seed}, step={case.step})", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return action_rows


def _ensure_matplotlib() -> None:
    global plt, Rectangle
    if plt is not None and Rectangle is not None:
        return
    import matplotlib.pyplot as plt_module
    from matplotlib.patches import Rectangle as rectangle_cls

    plt = plt_module
    Rectangle = rectangle_cls


def _plot_board(ax, board: np.ndarray, case: CaseSnapshot) -> None:
    colors = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            colors[r, c] = 0.15 if board[r, c] == "?" else 0.85
    ax.imshow(colors, cmap="Greys", vmin=0, vmax=1)
    ax.set_title("Visible State")
    _format_grid(ax)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            value = board[r, c]
            color = "white" if value == "?" else "black"
            ax.text(c, r, "?" if value == "?" else str(value), ha="center", va="center", color=color)
    _draw_logic_overlays(ax, case)


def _plot_q_heatmap(ax, q_grid: np.ndarray, board: np.ndarray, case: CaseSnapshot, model_name: str, best_action: int) -> None:
    im = ax.imshow(q_grid, cmap="viridis")
    ax.set_title(f"{model_name} Q-values")
    _format_grid(ax)
    best_row, best_col = action_to_coord(best_action, BOARD_SIZE)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if np.isnan(q_grid[r, c]):
                label = "" if board[r, c] == "?" else str(board[r, c])
                ax.text(c, r, label, ha="center", va="center", color="black", fontsize=8)
            else:
                ax.text(c, r, f"{q_grid[r, c]:.1f}", ha="center", va="center", color="white", fontsize=7)
    ax.scatter([best_col], [best_row], marker="*", s=220, facecolor="none", edgecolor="white", linewidth=1.8)
    _draw_logic_overlays(ax, case)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_probability_heatmap(ax, prob_grid: np.ndarray, board: np.ndarray, case: CaseSnapshot) -> None:
    im = ax.imshow(prob_grid, cmap="magma_r", vmin=0, vmax=1)
    ax.set_title("Estimated Mine Probability")
    _format_grid(ax)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if np.isnan(prob_grid[r, c]):
                label = "" if board[r, c] == "?" else str(board[r, c])
                ax.text(c, r, label, ha="center", va="center", color="black", fontsize=8)
            else:
                ax.text(c, r, f"{prob_grid[r, c]:.2f}", ha="center", va="center", color="white", fontsize=7)
    _draw_logic_overlays(ax, case)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _format_grid(ax) -> None:
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(BOARD_SIZE - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, BOARD_SIZE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)


def _draw_logic_overlays(ax, case: CaseSnapshot) -> None:
    for row, col in case.analysis.safe_cells:
        ax.add_patch(
            Rectangle(
                (col - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#39d353",
                linewidth=2.0,
            )
        )
    for row, col in case.analysis.mine_cells:
        ax.text(col, row, "X", ha="center", va="center", color="#ff4d4d", fontsize=14, fontweight="bold")


def write_action_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case_type",
        "seed",
        "step",
        "model",
        "selected_action",
        "selected_coord",
        "q_value",
        "mine_probability",
        "is_guaranteed_safe",
        "is_guaranteed_mine",
        "true_value",
        "used_exact_solver",
        "frontier_size",
        "valid_assignments",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_thesis_notes(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Catatan Teks Skripsi: Reasoning vs Probability",
        "",
        "## Kasus Reasoning",
        "",
        "Pada kasus reasoning, terdapat sedikitnya satu petak yang dapat dibuktikan aman berdasarkan constraint lokal Minesweeper. Petak aman tersebut ditandai dengan kotak hijau pada visualisasi. Jika nilai Q tertinggi agent berada pada petak tersebut, maka perilaku agent dapat diinterpretasikan sebagai reasoning-like behavior, yaitu kebijakan yang selaras dengan deduksi logis dari angka-angka yang sudah terbuka.",
        "",
        "## Kasus Probability",
        "",
        "Pada kasus probability, tidak terdapat petak yang dapat dibuktikan aman secara deterministik. Seluruh kandidat masih memiliki probabilitas mengandung ranjau lebih besar dari nol. Dalam kondisi ini, keputusan tidak dapat diselesaikan hanya dengan aturan logis lokal, sehingga agent harus memilih aksi berdasarkan preferensi risiko. Heatmap probabilitas menunjukkan estimasi risiko setiap petak, sedangkan heatmap Q-value menunjukkan preferensi aksi yang dipelajari oleh model.",
        "",
        "## Interpretasi",
        "",
        "Perbedaan kedua kasus ini memperjelas bahwa permainan Minesweeper tidak selalu dapat diselesaikan oleh algoritma deterministik sederhana. Ketika constraint menghasilkan petak aman, keputusan bersifat reasoning. Namun ketika tidak ada petak aman yang dapat dibuktikan, keputusan berubah menjadi problem probabilistik. Di sinilah pendekatan RL relevan, karena agent belajar mengurutkan aksi berdasarkan return jangka panjang di bawah ketidakpastian.",
        "",
        "## Selected Actions",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['case_type']} | {row['model']} memilih {row['selected_coord']} "
            f"dengan Q={row['q_value']} dan probabilitas ranjau={row['mine_probability']}."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative reasoning and probability Q-value cases."
    )
    parser.add_argument("--dqn-model", type=Path, default=PROJECT_ROOT / "dqn" / "dqn_best.keras")
    parser.add_argument("--caedqn-model", type=Path, default=PROJECT_ROOT / "caedqn" / "caedqn_best.keras")
    parser.add_argument("--max-seed", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--max-frontier", type=int, default=20)
    parser.add_argument("--preferred-model", choices=["DQN", "CAE-DQN"], default="CAE-DQN")
    parser.add_argument("--reasoning-output", type=Path, default=FIGURE_DIR / "reasoning_case.png")
    parser.add_argument("--probability-output", type=Path, default=FIGURE_DIR / "probability_case.png")
    parser.add_argument("--actions-output", type=Path, default=TABLE_DIR / "reasoning_probability_actions.csv")
    parser.add_argument("--thesis-notes-output", type=Path, default=TABLE_DIR / "reasoning_probability_thesis_notes_id.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    quiet_tensorflow_logs()

    if not args.dqn_model.exists():
        raise FileNotFoundError(f"DQN model not found: {args.dqn_model}")
    if not args.caedqn_model.exists():
        raise FileNotFoundError(f"CAE-DQN model not found: {args.caedqn_model}")

    print("Loading DQN and CAE-DQN models...")
    models = load_models(args.dqn_model, args.caedqn_model)

    print("Searching for reasoning/probability cases...")
    reasoning_case, probability_case = search_cases(
        args.max_seed,
        args.max_steps,
        args.max_frontier,
        models=models,
        preferred_model=args.preferred_model,
    )
    print(
        f"Reasoning case: seed={reasoning_case.seed}, step={reasoning_case.step}; "
        f"Probability case: seed={probability_case.seed}, step={probability_case.step}"
    )

    rows: list[dict] = []
    rows.extend(plot_case(reasoning_case, models, args.reasoning_output))
    rows.extend(plot_case(probability_case, models, args.probability_output))

    write_action_rows(args.actions_output, rows)
    write_thesis_notes(args.thesis_notes_output, rows)

    print(f"Saved reasoning case figure to {args.reasoning_output}")
    print(f"Saved probability case figure to {args.probability_output}")
    print(f"Saved selected-action table to {args.actions_output}")
    print(f"Saved Indonesian thesis notes to {args.thesis_notes_output}")


if __name__ == "__main__":
    main()
