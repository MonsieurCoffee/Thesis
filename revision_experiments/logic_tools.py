from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

import numpy as np

from common import BOARD_SIZE, N_MINES, coord_to_action


Coord = tuple[int, int]


@dataclass
class ConstraintAnalysis:
    safe_cells: set[Coord]
    mine_cells: set[Coord]
    constraints: list[tuple[set[Coord], int]]
    mine_probabilities: dict[Coord, float]
    frontier_size: int
    valid_assignments: int
    used_exact_solver: bool


def neighbors(row: int, col: int, rows: int = BOARD_SIZE, cols: int = BOARD_SIZE) -> list[Coord]:
    result: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
    return result


def state_value(state: list[dict], row: int, col: int, cols: int = BOARD_SIZE):
    return state[coord_to_action(row, col, cols)]["value"]


def visible_board_from_state(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
) -> np.ndarray:
    board = np.full((rows, cols), "?", dtype=object)
    for idx, tile in enumerate(state):
        row, col = divmod(idx, cols)
        value = tile["value"]
        board[row, col] = "?" if value == "U" else value
    return board


def unknown_cells(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
) -> set[Coord]:
    cells: set[Coord] = set()
    for idx, tile in enumerate(state):
        if tile["value"] == "U":
            cells.add(divmod(idx, cols))
    return cells


def numbered_cells(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
) -> list[tuple[Coord, int]]:
    cells: list[tuple[Coord, int]] = []
    for idx, tile in enumerate(state):
        value = tile["value"]
        if isinstance(value, (int, np.integer)) and 0 <= int(value) <= 8:
            cells.append((divmod(idx, cols), int(value)))
    return cells


def build_constraints(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
    known_mines: Iterable[Coord] | None = None,
) -> list[tuple[set[Coord], int]]:
    known_mines_set = set(known_mines or [])
    constraints: list[tuple[set[Coord], int]] = []
    for (row, col), number in numbered_cells(state, rows, cols):
        neigh = neighbors(row, col, rows, cols)
        unknown = {
            cell
            for cell in neigh
            if state_value(state, cell[0], cell[1], cols) == "U"
            and cell not in known_mines_set
        }
        flagged = sum(1 for cell in neigh if cell in known_mines_set)
        remaining = number - flagged
        if unknown:
            constraints.append((unknown, remaining))
    return constraints


def basic_deductions(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
    known_mines: Iterable[Coord] | None = None,
) -> tuple[set[Coord], set[Coord], list[tuple[set[Coord], int]]]:
    known_mines_set = set(known_mines or [])
    safe: set[Coord] = set()
    mines: set[Coord] = set()
    constraints = build_constraints(state, rows, cols, known_mines_set)

    for cells, remaining in constraints:
        if remaining == 0:
            safe.update(cells)
        elif remaining == len(cells):
            mines.update(cells)

    safe.difference_update(known_mines_set)
    mines.difference_update(safe)
    return safe, mines, constraints


def iterative_basic_deductions(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
    known_mines: Iterable[Coord] | None = None,
) -> tuple[set[Coord], set[Coord], list[tuple[set[Coord], int]]]:
    safe_total: set[Coord] = set()
    mines_total: set[Coord] = set(known_mines or [])
    constraints: list[tuple[set[Coord], int]] = []

    changed = True
    while changed:
        safe, mines, constraints = basic_deductions(state, rows, cols, mines_total)
        new_safe = safe - safe_total
        new_mines = mines - mines_total
        safe_total.update(new_safe)
        mines_total.update(new_mines)
        changed = bool(new_safe or new_mines)

    mines_total.difference_update(set(known_mines or []))
    return safe_total, mines_total, constraints


def exact_frontier_probabilities(
    state: list[dict],
    rows: int = BOARD_SIZE,
    cols: int = BOARD_SIZE,
    n_mines: int = N_MINES,
    known_mines: Iterable[Coord] | None = None,
    max_frontier: int = 20,
) -> ConstraintAnalysis:
    known_mines_set = set(known_mines or [])
    all_unknown = unknown_cells(state, rows, cols) - known_mines_set
    constraints = build_constraints(state, rows, cols, known_mines_set)
    frontier = sorted(set().union(*(cells for cells, _ in constraints)) if constraints else set())

    safe_basic, mine_basic, _ = iterative_basic_deductions(
        state, rows, cols, known_mines_set
    )

    if not all_unknown:
        return ConstraintAnalysis(
            safe_cells=set(),
            mine_cells=set(),
            constraints=constraints,
            mine_probabilities={},
            frontier_size=0,
            valid_assignments=0,
            used_exact_solver=True,
        )

    remaining_global = max(0, n_mines - len(known_mines_set))

    if not constraints:
        probability = remaining_global / len(all_unknown)
        probabilities = {cell: probability for cell in all_unknown}
        return ConstraintAnalysis(
            safe_cells={cell for cell, prob in probabilities.items() if prob == 0.0},
            mine_cells={cell for cell, prob in probabilities.items() if prob == 1.0},
            constraints=constraints,
            mine_probabilities=probabilities,
            frontier_size=0,
            valid_assignments=1,
            used_exact_solver=True,
        )

    if len(frontier) > max_frontier:
        probabilities = _local_probability_fallback(
            all_unknown, constraints, remaining_global
        )
        return ConstraintAnalysis(
            safe_cells=safe_basic,
            mine_cells=mine_basic,
            constraints=constraints,
            mine_probabilities=probabilities,
            frontier_size=len(frontier),
            valid_assignments=0,
            used_exact_solver=False,
        )

    frontier_index = {cell: idx for idx, cell in enumerate(frontier)}
    off_frontier = sorted(all_unknown - set(frontier))
    weighted_total = 0
    mine_weight = {cell: 0 for cell in all_unknown}
    valid_assignments = 0

    for mask in range(1 << len(frontier)):
        assignment_mines: set[Coord] = set()
        for idx, cell in enumerate(frontier):
            if mask & (1 << idx):
                assignment_mines.add(cell)

        valid = True
        for cells, remaining in constraints:
            count = sum(1 for cell in cells if cell in assignment_mines)
            if count != remaining:
                valid = False
                break
        if not valid:
            continue

        mines_in_frontier = len(assignment_mines)
        off_mines = remaining_global - mines_in_frontier
        if off_mines < 0 or off_mines > len(off_frontier):
            continue

        weight = comb(len(off_frontier), off_mines) if off_frontier else 1
        valid_assignments += 1
        weighted_total += weight
        for cell in assignment_mines:
            mine_weight[cell] += weight
        if off_frontier and off_mines:
            for cell in off_frontier:
                mine_weight[cell] += weight * (off_mines / len(off_frontier))

    if weighted_total == 0:
        probabilities = _local_probability_fallback(
            all_unknown, constraints, remaining_global
        )
        return ConstraintAnalysis(
            safe_cells=safe_basic,
            mine_cells=mine_basic,
            constraints=constraints,
            mine_probabilities=probabilities,
            frontier_size=len(frontier),
            valid_assignments=0,
            used_exact_solver=False,
        )

    probabilities = {
        cell: float(mine_weight[cell] / weighted_total)
        for cell in all_unknown
    }
    safe = {cell for cell, prob in probabilities.items() if np.isclose(prob, 0.0)}
    mines = {cell for cell, prob in probabilities.items() if np.isclose(prob, 1.0)}

    return ConstraintAnalysis(
        safe_cells=safe,
        mine_cells=mines,
        constraints=constraints,
        mine_probabilities=probabilities,
        frontier_size=len(frontier),
        valid_assignments=valid_assignments,
        used_exact_solver=True,
    )


def _local_probability_fallback(
    all_unknown: set[Coord],
    constraints: list[tuple[set[Coord], int]],
    remaining_global: int,
) -> dict[Coord, float]:
    default_probability = remaining_global / max(1, len(all_unknown))
    probabilities: dict[Coord, float] = {}
    for cell in all_unknown:
        local_scores = [
            remaining / len(cells)
            for cells, remaining in constraints
            if cell in cells and cells
        ]
        if local_scores:
            probabilities[cell] = float(max(0.0, min(1.0, max(local_scores))))
        else:
            probabilities[cell] = float(max(0.0, min(1.0, default_probability)))
    return probabilities
