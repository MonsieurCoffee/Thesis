import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Disable oneDNN custom ops warnings
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '3'       # Suppress VLOG messages
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'       # Additional VLOG suppression
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'          # Suppress absl logs (if available)

import csv
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model #type:ignore
from minesweeper_env import MinesweeperEnv

MODEL_PATH = "dqn_best.keras"               # Model Fetch
BOARD_SIZE = 6                              # Board Size
N_MINES = 4                                 # Mines 
EVAL_SEEDS = [1, 2, 3, 4, 5]                # 5 Boards
EPISODES_PER_SEED = 500                     # Tested on 500 Episodes / Board
ALGORITHM_NAME = "DQN"
OUTPUT_CSV = "csv/eval_dqn_4mines.csv"

class EvalAgent:
    # Initialize Model & Environment
    def __init__(self, model, env):
        self.model = model
        self.env = env

    # Action Chosen
    def get_action(self, state):
        # One-Hot Board Input
        unknown_channel = state[:, :, 9]
        flat_unknown = unknown_channel.reshape(-1)
        unsolved = [i for i, u in enumerate(flat_unknown) if u == 1.0]

        if not unsolved:
            return 0
        
        # Pure Greedy Policy
        input_batch = np.expand_dims(state, axis=0)
        outputs = self.model.predict_on_batch(input_batch)
        q_values = outputs[0]
        mask = (flat_unknown == 0)
        q_values[mask] = np.min(q_values)
        return int(np.argmax(q_values))

def main():
    # Debugging Purposes
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    # Save Evaluation CSV
    os.makedirs("csv", exist_ok=True)
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['algorithm', 'board_size', 'n_mines', 'eval_seed', 'episode', 'win', 'progress', 'total_reward', 'n_clicks'])

    total_episodes = len(EVAL_SEEDS) * EPISODES_PER_SEED
    pbar = tqdm(total=total_episodes, desc="Evaluating DQN", unit="ep")

    # 5 Seeds Loop
    for eval_seed in EVAL_SEEDS:
        env = MinesweeperEnv(BOARD_SIZE, BOARD_SIZE, N_MINES, seed=eval_seed)
        agent = EvalAgent(model, env)

        # 500 Episodes Loop
        for episode in range(1, EPISODES_PER_SEED + 1):
            # Game Start Initialization
            env.reset()
            episode_reward = 0.0
            done = False
            n_clicks = 0

            # Loop Until Episode Ends
            while not done:
                state = env.state_im
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                n_clicks += 1

            win = 1 if env.n_wins > 0 else 0
            progress = env.n_progress

            # Write Metrics to CSV
            csv_writer.writerow([
                ALGORITHM_NAME, BOARD_SIZE, N_MINES, eval_seed, episode,
                win, progress, round(episode_reward, 2), n_clicks
            ])
            pbar.update(1)

        csv_file.flush()

    pbar.close()
    csv_file.close()
    print(f"\nEvaluation completed. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()