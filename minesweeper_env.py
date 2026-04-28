# minesweeper_env.py
import random
import numpy as np

class MinesweeperEnv:
    # Initialize Board
    def __init__(self, width, height, n_mines, seed=None, rewards=None):
        # Storing Board Dimensions
        self.n_rows = width
        self.n_cols = height
        self.n_tiles = self.n_rows * self.n_cols
        self.n_mines = n_mines

        # Default Reward Structure
        if rewards is None:
            self.rewards = {
                'win': 36,
                'lose': -36,
                'progress': 1.0,
            }
        else:
            self.rewards = rewards  # Manual Reward Input

        # Internal Data Containers
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=object)              # Hidden Mine Layout
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=object)             # Computed Number Layout
        self.state = []                                                             # Internal State Memory
        self.state_im = np.zeros((self.n_rows, self.n_cols, 10), dtype=np.float32)  # One-Hot Encoded Image

        # Counters
        self.n_clicks = 0       # Agent Clicks
        self.n_progress = 0     # Progress
        self.n_wins = 0         # Boolean Win Condition

        # Manual Seed Input
        if seed is not None:
            self.set_seed(seed)

        # Initialize Board and Initial State
        self.reset()

    # Board Seed
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # Resets Board (New Game)
    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0
        self._init_grid()
        self._compute_board()
        self._init_state()
        return self.state_im

    # Mine Placement
    def _init_grid(self):
        self.grid.fill(None)
        mines_placed = 0
        while mines_placed < self.n_mines:
            r = random.randint(0, self.n_rows - 1)
            c = random.randint(0, self.n_cols - 1)
            if self.grid[r, c] != 'B':
                self.grid[r, c] = 'B'
                mines_placed += 1

    # Mine Surveyor
    def _get_neighbors(self, row, col):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                    neighbors.append(self.grid[nr, nc])
        return neighbors

    # Number Counting
    def _count_adjacent_mines(self, row, col):
        neighbors = self._get_neighbors(row, col)
        return sum(1 for v in neighbors if v == 'B')

    # Number Placement
    def _compute_board(self):
        self.board = self.grid.copy()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.board[r, c] != 'B':
                    self.board[r, c] = self._count_adjacent_mines(r, c)

    # Wipes Board Clean
    def _init_state(self):
        self.state = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                self.state.append({'coord': (r, c), 'value': 'U'})
        self._update_state_image()

    # One-Hot Encoding Logic
    def _one_hot_cell(self, value):
        one_hot = [0.0] * 10
        if value == 'U':
            one_hot[9] = 1.0
        elif isinstance(value, (int, float)) and 0 <= value <= 8:
            one_hot[int(value)] = 1.0
        else:
            one_hot[9] = 1.0
        return one_hot

    # Update Internal State Memory and Converts Board Into One-Hot Representation
    def _update_state_image(self):
        im = np.zeros((self.n_rows, self.n_cols, 10), dtype=np.float32)
        for idx, tile in enumerate(self.state):
            r = idx // self.n_cols
            c = idx % self.n_cols
            vec = self._one_hot_cell(tile['value'])
            im[r, c, :] = vec
        self.state_im = im

    # Reveal Clicked Cell
    def _reveal_cell(self, row, col):
        idx = row * self.n_cols + col
        self.state[idx]['value'] = self.board[row, col]

    # Recursive Reveal (DFS)
    def _reveal_neighbors(self, row, col, processed):
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

    # Click Consequence
    def click(self, action_index):
        # Reveals Number or Bomb
        r = action_index // self.n_cols
        c = action_index % self.n_cols
        value = self.board[r, c]

        # Redirect First Click to Random Safe Tile
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

        if value == 0:
            self._reveal_neighbors(r, c, [])

        self.n_clicks += 1

    # Game Process Logic
    def step(self, action_index):
        # Count Unknown Tiles
        prev_unknown = np.sum(self.state_im[:, :, 9] == 1)

        # Board Coords & True Board
        r = action_index // self.n_cols
        c = action_index % self.n_cols
        neighbors = self._get_neighbors(r, c)

        self.click(action_index)    # Performs Click Action
        self._update_state_image()  # Rebuilds New State

        # Default Return Values
        done = False
        reward = 0

        # Termination Condition
        if self.state[action_index]['value'] == 'B':
            reward = self.rewards['lose']
            done = True
        
        # Win condition
        elif np.sum(self.state_im[:, :, 9] == 1) == self.n_mines:
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        # Progress
        else:
            reward = self.rewards['progress']
            self.n_progress += 1

        return self.state_im, reward, done

    # Visual Debugging Purposes
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