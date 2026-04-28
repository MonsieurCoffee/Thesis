# game_demo.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import tensorflow as tf
from minesweeper_env import MinesweeperEnv

# ==================== CONFIGURATION ====================
AGENT_TYPE = 'dqn'                     # 'dqn' or 'caedqn'
MODEL_PATH = 'dqn/dqn_best.keras'   # path to model
NUM_EPISODES = 50                         # total episodes to run
AUTO_ADVANCE = True                       # True = auto next episode, False = press Enter to continue
# =======================================================

# Environment settings
TIME = 0.01
BASE_SEED = 300          # None = truly random boards each run
ROWS, COLS, MINES = 6, 6, 4

# Create environment ONCE (same as training)
env = MinesweeperEnv(width=ROWS, height=COLS, n_mines=MINES, seed=BASE_SEED)

print(f"Loading {AGENT_TYPE} model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Helper for action selection (unchanged) ---
def get_action(env, state_im, valid_actions):
    state_batch = np.expand_dims(state_im, axis=0)
    outputs = model.predict(state_batch, verbose=0)
    if isinstance(outputs, (list, tuple)):
        q_values = outputs[0][0]
    else:
        q_values = outputs[0]
    mask = np.ones(env.n_tiles, dtype=bool)
    mask[valid_actions] = False
    q_values[mask] = -np.inf
    return int(np.argmax(q_values))

# --- Rendering functions (unchanged) ---
def render_board(ax, data_grid, title, highlight=None):
    colors = {
        '?': 'lightgray', '*': 'red', 'B': 'red',
        '0': 'white', '1': 'blue', '2': 'green', '3': 'red',
        '4': 'darkblue', '5': 'darkred', '6': 'cyan', '7': 'black', '8': 'gray'
    }
    rows, cols = data_grid.shape
    dummy = np.zeros((rows, cols))
    ax.clear()
    ax.pcolormesh(dummy, edgecolors='black', linewidth=1, cmap='Greys', vmin=0, vmax=1)
    for r in range(rows):
        for c in range(cols):
            val = data_grid[r, c]
            char = str(val) if val != 'B' else '*'
            color = colors.get(char, 'black')
            ax.text(c + 0.5, r + 0.5, char, ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    if highlight is not None:
        r, c = highlight
        rect = Rectangle((c, r), 1, 1, linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    ax.set_title(title)

def get_revealed_grid(env):
    grid = np.full((env.n_rows, env.n_cols), '?', dtype=object)
    for idx, tile in enumerate(env.state):
        r, c = divmod(idx, env.n_cols)
        val = tile['value']
        grid[r, c] = val if val != 'U' else '?'
    return grid

def get_true_grid(env):
    return env.board.copy()

# --- Replay controller (unchanged) ---
class EpisodeReplay:
    def __init__(self, fig, ax1, ax2, env, history_revealed, history_actions):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.env = env
        self.history_revealed = history_revealed
        self.history_actions = history_actions
        self.current_step = len(history_revealed) - 1
        self.next_episode = False

    def update_display(self):
        step = self.current_step
        total = len(self.history_revealed) - 1
        title = "Revealed (Initial)" if step == 0 else f"Revealed (Step {step}/{total})"
        render_board(self.ax1, self.history_revealed[step], title)
        if step > 0:
            r, c = self.history_actions[step-1]
            render_board(self.ax2, get_true_grid(self.env), f"True Board (Action: ({r},{c}))")
        else:
            render_board(self.ax2, get_true_grid(self.env), "True Board (Hidden)")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'right' and self.current_step < len(self.history_revealed) - 1:
            self.current_step += 1
            self.update_display()
        elif event.key == 'left' and self.current_step > 0:
            self.current_step -= 1
            self.update_display()
        elif event.key == 'enter':
            self.next_episode = True
            plt.close(self.fig)

# --- Main loop ---
wins = 0
print(f"\nRunning {NUM_EPISODES} episodes on seed {BASE_SEED}...")
print(f"Auto-advance: {'ON' if AUTO_ADVANCE else 'OFF'}\n")

for episode in range(1, NUM_EPISODES + 1):
    print(f"\n========== Episode {episode}/{NUM_EPISODES} ==========")
    env.reset()

    history_revealed = [get_revealed_grid(env)]
    history_actions = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    render_board(ax1, history_revealed[0], "Revealed (Agent View)")
    render_board(ax2, get_true_grid(env), "True Board (Hidden)")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)   # shorter initial pause

    click_count = 0
    done = False
    while not done:
        unknown = [i for i, t in enumerate(env.state) if t['value'] == 'U']
        if not unknown:
            print("No unknown cells left (win?)")
            break

        action = get_action(env, env.state_im, unknown)
        r, c = divmod(action, env.n_cols)
        click_count += 1
        print(f"Click {click_count}: ({r}, {c})")

        # Highlight (brief)
        render_board(ax1, get_revealed_grid(env), "Revealed (Agent View)", highlight=(r, c))
        render_board(ax2, get_true_grid(env), "True Board (Hidden)", highlight=(r, c))
        plt.tight_layout()
        plt.draw()
        plt.pause(TIME)

        _, reward, done = env.step(action)
        print(f"Reward: {reward:.1f}")

        history_revealed.append(get_revealed_grid(env))
        history_actions.append((r, c))

        render_board(ax1, history_revealed[-1], f"Revealed (Step {click_count})")
        render_board(ax2, get_true_grid(env), "True Board (Hidden)")
        plt.tight_layout()
        plt.draw()

        if done:
            if reward > 0:
                wins += 1
                print("Agent won!")
            else:
                print("Mine hit! Game over.")
            plt.pause(1.0)   # shorter end‑of‑episode pause
            break
        else:
            plt.pause(0.2)   # shorter between‑moves pause

    # --- Post‑episode: replay only if AUTO_ADVANCE is False ---
    if not AUTO_ADVANCE:
        plt.ioff()
        replay = EpisodeReplay(fig, ax1, ax2, env, history_revealed, history_actions)
        cid = fig.canvas.mpl_connect('key_press_event', replay.on_key)
        replay.update_display()

        print("\n--- Episode finished ---")
        print("Use LEFT/RIGHT arrow keys to replay moves.")
        print("Press ENTER to start next episode (or close window to exit).")

        while not replay.next_episode:
            if not plt.fignum_exists(fig.number):
                print("Window closed. Exiting.")
                exit()
            plt.pause(0.1)

        fig.canvas.mpl_disconnect(cid)
    else:
        # In auto mode, just show the final board briefly and close the figure
        plt.ioff()
        print("\n--- Episode finished (auto‑advancing) ---")
        # Keep window open for a moment so the user can see the result
        plt.pause(0.5)

    plt.close(fig)

print(f"\n===== Summary =====")
print(f"Total episodes: {NUM_EPISODES}")
print(f"Wins: {wins}")
print(f"Win rate: {wins / NUM_EPISODES * 100:.1f}%")