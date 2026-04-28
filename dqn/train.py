import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Disable oneDNN custom ops warnings
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '3'       # Suppress VLOG messages
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'       # Additional VLOG suppression
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'          # Suppress absl logs (if available)

import csv
import time
import numpy as np
from minesweeper_env import MinesweeperEnv
from dqn_agent import DQNAgent, DISCOUNT_FACTOR

WIDTH = 6           # Columns (c)
HEIGHT = 6          # Rows (r)
N_MINES = 4         # Number of Mines
EPISODES = 80_000   # Training Episodes
ENV_SEED = 2004     # Board Seed (Training)
MODEL_SEED = 123    # Model Seed (Initial)
PRINT_EVERY = 1000  # Average Evaluation
MODEL_NAME = "dqn"

def main():
    # Debugging Purposes
    print("=" * 60)
    print(f"Algorithm: Regular DQN")
    print(f"Environment seed: {ENV_SEED}")
    print(f"Model seed: {MODEL_SEED}")
    print(f"Board size: {WIDTH}x{HEIGHT}, Mines: {N_MINES}")
    print(f"Total episodes: {EPISODES}")
    print("=" * 60)

    # Agent & Environment Initialization
    env = MinesweeperEnv(WIDTH, HEIGHT, N_MINES, seed=ENV_SEED)
    agent = DQNAgent(env, model_name=MODEL_NAME, seed=MODEL_SEED)

    # Save Training CSV
    os.makedirs("csv", exist_ok=True)
    csv_filename = "csv/training_dqn.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'total_reward', 'progress', 'win', 'n_clicks', 'epsilon', 'learn_rate'])

    # Save Evaluation CSV (1000 Episodeds)
    eval_csv_filename = "csv/training_eval_dqn.csv"
    eval_csv_file = open(eval_csv_filename, 'w', newline='')
    eval_csv_writer = csv.writer(eval_csv_file)
    eval_csv_writer.writerow(['episode', 'med_progress', 'win_rate', 'med_reward', 'epsilon', 'learn_rate', 'time_elapsed'])

    # Metrics Initialization
    episode_rewards = []
    episode_progress = []
    episode_wins = []

    # Timer
    start_time = time.time()

    # Best Model Naming (Highest Win Rate)
    best_win_rate = -1.0
    best_model_path = f"{MODEL_NAME}_best.keras"

    # Training Loop
    for episode in range(1, EPISODES + 1):
        env.reset()         # Initialize Board [minesweeper_env.py]
        episode_reward = 0
        done = False

        # Loop Until Episode Ends
        while not done:
            current_state = env.state_im                                                    # Agent Observation [minesweeper_env.py]
            action = agent.get_action(current_state)                                        # Agent Action [dqn_agent.py]
            new_state, reward, done = env.step(action)                                      # Feedback [minesweeper_env.py]
            episode_reward += reward                                                        # Total Reward Counter (1 Episode)
            agent.update_replay_memory((current_state, action, reward, new_state, done))    # Store Transition to Replay Buffer [dqn_agent.py]
            agent.train(done)

        win = 1 if env.n_wins > 0 else 0        # Episode Binary Win Flag

        # Append Reward, Progress, and Wins to List for 1000 Episodes Eval
        episode_rewards.append(episode_reward)
        episode_progress.append(env.n_progress)
        episode_wins.append(win)

        # Write Metrics to Training CSV
        csv_writer.writerow([
            episode,
            round(episode_reward, 2),
            env.n_progress,
            win,
            env.n_clicks,
            round(agent.epsilon, 4),
            round(agent.learning_rate, 6)
        ])
        csv_file.flush()

        # Print 1000 Episodes Evaluation
        if episode % PRINT_EVERY == 0:
            med_progress = round(np.median(episode_progress[-PRINT_EVERY:]), 2)
            win_rate = round(np.sum(episode_wins[-PRINT_EVERY:]) / PRINT_EVERY, 2)
            med_reward = round(np.median(episode_rewards[-PRINT_EVERY:]), 2)
            elapsed = time.time() - start_time

            print(f"Episode {episode}/{EPISODES} | "
                  f"Med progress: {med_progress} | "
                  f"Win rate: {win_rate} | "
                  f"Med reward: {med_reward} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"LR: {agent.learning_rate:.6f} | "
                  f"Time: {elapsed:.2f}s")

            eval_csv_writer.writerow([
                episode,
                med_progress,
                win_rate,
                med_reward,
                round(agent.epsilon, 4),
                round(agent.learning_rate, 6),
                round(elapsed, 2)
            ])
            eval_csv_file.flush()

            # Save Best Model Logic
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.model.save(best_model_path)
                print(f"  New best model saved (win rate {win_rate:.2f})")

    # Debuggin Purposes
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best model saved to {best_model_path} (win rate {best_win_rate:.2f})")
    print(f"Training data saved to {csv_filename}")
    print(f"Evaluation snapshots saved to {eval_csv_filename}")

    csv_file.close()
    eval_csv_file.close()

if __name__ == "__main__":
    main()