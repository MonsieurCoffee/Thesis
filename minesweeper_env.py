# minesweeper_env.py
import random
import numpy as np

class MinesweeperEnv:
    """
    Minesweeper environment for DQN training.
    Uses one‑hot encoding (10 channels) as described in Wang et al. (2025).
    Reward structure: win=+36, lose=-36, progress=+1, guess=-0.5, no_progress=-0.5.
    """
    def __init__(self, width, height, n_mines, seed=None, rewards=None):
        self.n_rows = width
        self.n_cols = height
        self.n_tiles = self.n_rows * self.n_cols
        self.n_mines = n_mines

        # Default reward structure (scaled for 6x6 board, 36 cells)
        if rewards is None:
            self.rewards = {
                'win': 36,
                'lose': -36,
                'progress': 1.0,
            }
        else:
            self.rewards = rewards

        # Internal state
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=object)
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=object)
        self.state = []               # list of dicts with 'coord' and 'value'
        self.state_im = np.zeros((self.n_rows, self.n_cols, 10), dtype=np.float32)
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0

        if seed is not None:
            self.set_seed(seed)

        self.reset()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """Reset the environment and return the initial one‑hot state."""
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0
        self._init_grid()
        self._compute_board()
        self._init_state()
        return self.state_im   # shape (n_rows, n_cols, 10)

    def _init_grid(self):
        """Place mines randomly."""
        self.grid.fill(None)
        mines_placed = 0
        while mines_placed < self.n_mines:
            r = random.randint(0, self.n_rows - 1)
            c = random.randint(0, self.n_cols - 1)
            if self.grid[r, c] != 'B':
                self.grid[r, c] = 'B'
                mines_placed += 1

    def _get_neighbors(self, row, col):
        """Return list of neighbour values from the grid."""
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                    neighbors.append(self.grid[nr, nc])
        return neighbors

    def _count_adjacent_mines(self, row, col):
        neighbors = self._get_neighbors(row, col)
        return sum(1 for v in neighbors if v == 'B')

    def _compute_board(self):
        """Build board with numbers 0‑8 for non‑mine cells."""
        self.board = self.grid.copy()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.board[r, c] != 'B':
                    self.board[r, c] = self._count_adjacent_mines(r, c)

    def _init_state(self):
        """Initialise state list with all 'U' (unknown)."""
        self.state = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                self.state.append({'coord': (r, c), 'value': 'U'})
        self._update_state_image()

    def _one_hot_cell(self, value):
        """
        Convert a cell value to a 10‑channel one‑hot vector.
        Channels 0‑8: numbers 0‑8 (1 if matching, else 0)
        Channel 9: 1 if unknown, else 0
        """
        one_hot = [0.0] * 10
        if value == 'U':
            one_hot[9] = 1.0
        elif isinstance(value, (int, float)) and 0 <= value <= 8:
            one_hot[int(value)] = 1.0
        # Mines ('B') should never appear in state_im because they are revealed as 'B' only upon loss.
        # If they appear accidentally, treat as unknown (safe fallback).
        else:
            one_hot[9] = 1.0
        return one_hot

    def _update_state_image(self):
        """Build the one‑hot state image from the state list."""
        im = np.zeros((self.n_rows, self.n_cols, 10), dtype=np.float32)
        for idx, tile in enumerate(self.state):
            r = idx // self.n_cols
            c = idx % self.n_cols
            vec = self._one_hot_cell(tile['value'])
            im[r, c, :] = vec
        self.state_im = im

    def _reveal_cell(self, row, col):
        """Reveal a single cell (set its value in state list)."""
        idx = row * self.n_cols + col
        self.state[idx]['value'] = self.board[row, col]

    def _reveal_neighbors(self, row, col, processed):
        """Recursively reveal all zero‑value neighbours."""
        processed.append((row, col))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols and
                        (nr, nc) not in processed):
                    self._reveal_cell(nr, nc)
                    if self.board[nr, nc] == 0:
                        self._reveal_neighbors(nr, nc, processed)

    def click(self, action_index):
        """Perform a click at the given action index (0..n_tiles-1)."""
        r = action_index // self.n_cols
        c = action_index % self.n_cols
        value = self.board[r, c]

        # First move safety: avoid mine on first click
        if value == 'B' and self.n_clicks == 0:
            safe_indices = []
            for i in range(self.n_tiles):
                rr = i // self.n_cols
                cc = i % self.n_cols
                if self.board[rr, cc] != 'B':
                    safe_indices.append(i)
            new_idx = random.choice(safe_indices)
            r = new_idx // self.n_cols
            c = new_idx % self.n_cols
            value = self.board[r, c]
            self._reveal_cell(r, c)
        else:
            self._reveal_cell(r, c)

        # If the revealed cell is 0, reveal neighbours recursively
        if value == 0:
            self._reveal_neighbors(r, c, [])

        self.n_clicks += 1

    def step(self, action_index):
        """
        Take an action, return (next_state, reward, done).
        next_state is the one‑hot image.
        """
        # Keep a copy of the previous unknown count for progress detection
        prev_unknown = np.sum(self.state_im[:, :, 9] == 1)

        r = action_index // self.n_cols
        c = action_index % self.n_cols
        neighbors = self._get_neighbors(r, c)

        self.click(action_index)
        self._update_state_image()

        done = False
        reward = 0

        # Loss: clicked on a mine
        if self.state[action_index]['value'] == 'B':
            reward = self.rewards['lose']
            done = True

        # Win: all unknown cells are mines
        elif np.sum(self.state_im[:, :, 9] == 1) == self.n_mines:
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        # No progress (unknown count unchanged)
        elif np.sum(self.state_im[:, :, 9] == 1) == prev_unknown:
            reward = self.rewards['no_progress']

        # Progress made
        else:
            # Guess? All neighbours are unknown (value 'U')
            if all(n == 'U' for n in neighbors):
                reward = self.rewards['guess']
            else:
                reward = self.rewards['progress']
                self.n_progress += 1

        return self.state_im, reward, done

    # Optional: simple text display for debugging
    def draw_state(self):
        grid_display = np.full((self.n_rows, self.n_cols), '?', dtype=str)
        for idx, tile in enumerate(self.state):
            r = idx // self.n_cols
            c = idx % self.n_cols
            val = tile['value']
            if val == 'U':
                grid_display[r, c] = '?'
            elif val == 'B':
                grid_display[r, c] = 'B'
            else:
                grid_display[r, c] = str(val)
        print(grid_display)