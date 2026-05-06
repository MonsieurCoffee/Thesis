from __future__ import annotations

import numpy as np

from common import action_to_coord, coord_to_action, unknown_indices_from_image
from logic_tools import (
    Coord,
    iterative_basic_deductions,
)


class RandomAgent:
    """Uniform random baseline over legal unrevealed cells."""

    name = "Random Picker"

    def __init__(self, env, seed: int | None = None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.logical_moves = 0
        self.forced_guesses = 0

    def reset_episode(self) -> None:
        self.logical_moves = 0
        self.forced_guesses = 0

    def get_action(self, state_im):
        candidates = unknown_indices_from_image(state_im)
        if not candidates:
            return 0
        self.forced_guesses += 1
        return int(self.rng.choice(candidates))


class RuleBasedDeterministicAgent:
    """
    Deterministic Minesweeper baseline.

    It applies local constraints only:
    - if all remaining adjacent unknowns must be mines, mark them as mines;
    - if all adjacent mines have already been identified, open the rest.

    When no guaranteed safe cell exists, it performs a fixed deterministic guess
    from the remaining unrevealed cells. The hidden board is never accessed.
    """

    name = "Rule-Based Deterministic"

    def __init__(self, env, initial_strategy: str = "center"):
        self.env = env
        self.initial_strategy = initial_strategy
        self.known_mines: set[Coord] = set()
        self.logical_moves = 0
        self.forced_guesses = 0

    def reset_episode(self) -> None:
        self.known_mines = set()
        self.logical_moves = 0
        self.forced_guesses = 0

    def analyze_state(self):
        safe, mines, constraints = iterative_basic_deductions(
            self.env.state,
            self.env.n_rows,
            self.env.n_cols,
            self.known_mines,
        )
        self.known_mines.update(mines)
        safe.difference_update(self.known_mines)
        return safe, self.known_mines.copy(), constraints

    def get_action(self, state_im):
        candidates = set(
            action_to_coord(idx, self.env.n_cols)
            for idx in unknown_indices_from_image(state_im)
        )
        if not candidates:
            return 0

        if len(candidates) == self.env.n_tiles:
            action = self._initial_action(candidates)
            self.forced_guesses += 1
            return action

        safe, mines, _ = self.analyze_state()
        self.known_mines.update(mines)
        safe_candidates = sorted(safe & candidates)
        if safe_candidates:
            self.logical_moves += 1
            row, col = safe_candidates[0]
            return coord_to_action(row, col, self.env.n_cols)

        self.forced_guesses += 1
        row, col = self._deterministic_guess(candidates)
        return coord_to_action(row, col, self.env.n_cols)

    def _initial_action(self, candidates: set[Coord]) -> int:
        if self.initial_strategy == "corner":
            row, col = 0, 0
        else:
            row, col = self.env.n_rows // 2, self.env.n_cols // 2
        if (row, col) not in candidates:
            row, col = sorted(candidates)[0]
        return coord_to_action(row, col, self.env.n_cols)

    def _deterministic_guess(self, candidates: set[Coord]) -> Coord:
        available = sorted(cell for cell in candidates if cell not in self.known_mines)
        if available:
            return available[0]
        return sorted(candidates)[0]
