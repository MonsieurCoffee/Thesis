# caedqn/train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import time
import numpy as np
import tensorflow as tf
from minesweeper_env import MinesweeperEnv
from caedqn_agent import CAEDQNAgent, BETA, RIDGE

# ========== Training Hyperparameters ==========
WIDTH = 6
HEIGHT = 6
N_MINES = 4
EPISODES = 80_000
ENV_SEED = 2004
MODEL_SEED = 123
PRINT_EVERY = 1000
MODEL_NAME = "caedqn"

def main():
    np.random.seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)

    print("=" * 60)
    print("Algorithm: CAE-DQN (Critic as Explorer)")
    print(f"Environment seed: {ENV_SEED}")
    print(f"Model seed: {MODEL_SEED}")
    print(f"Board size: {WIDTH}x{HEIGHT}, Mines: {N_MINES}")
    print(f"Beta (exploration coefficient): {BETA}")
    print(f"Ridge (regularisation): {RIDGE}")
    print(f"Total episodes: {EPISODES}")
    print("=" * 60)

    env = MinesweeperEnv(WIDTH, HEIGHT, N_MINES, seed=ENV_SEED)
    agent = CAEDQNAgent(env, model_name=MODEL_NAME, seed=MODEL_SEED)

    os.makedirs("csv", exist_ok=True)
    csv_filename = "csv/training_caedqn.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'total_reward', 'progress', 'win', 'n_clicks', 'beta', 'learn_rate'])

    eval_csv_filename = "csv/training_eval_caedqn.csv"
    eval_csv_file = open(eval_csv_filename, 'w', newline='')
    eval_csv_writer = csv.writer(eval_csv_file)
    eval_csv_writer.writerow(['episode', 'med_progress', 'win_rate', 'med_reward', 'beta', 'learn_rate', 'time_elapsed'])

    episode_rewards = []
    episode_progress = []
    episode_wins = []

    start_time = time.time()
    best_win_rate = -1.0
    best_model_path = f"{MODEL_NAME}_best.keras"

    for episode in range(1, EPISODES + 1):
        env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_state = env.state_im
            action = agent.get_action(current_state)
            new_state, reward, done = env.step(action)

            # --- CAE exploration bonus ---
            bonus = agent.compute_bonus(new_state) if not done else 0
            augmented_reward = reward + bonus
            # -----------------------------

            episode_reward += augmented_reward   # track augmented reward
            agent.update_replay_memory((current_state, action, augmented_reward, new_state, done))
            agent.train(done)

        win = 1 if env.n_wins > 0 else 0
        episode_rewards.append(episode_reward)
        episode_progress.append(env.n_progress)
        episode_wins.append(win)

        csv_writer.writerow([
            episode,
            round(episode_reward, 2),
            env.n_progress,
            win,
            env.n_clicks,
            round(agent.beta, 4),
            round(agent.learning_rate, 6)
        ])
        csv_file.flush()

        if episode % PRINT_EVERY == 0:
            med_progress = round(np.median(episode_progress[-PRINT_EVERY:]), 2)
            win_rate = round(np.sum(episode_wins[-PRINT_EVERY:]) / PRINT_EVERY, 2)
            med_reward = round(np.median(episode_rewards[-PRINT_EVERY:]), 2)
            elapsed = time.time() - start_time

            print(f"Episode {episode}/{EPISODES} | "
                  f"Med progress: {med_progress} | "
                  f"Win rate: {win_rate} | "
                  f"Med reward: {med_reward} | "
                  f"Beta: {agent.beta:.4f} | "
                  f"LR: {agent.learning_rate:.6f} | "
                  f"Time: {elapsed:.2f}s")

            eval_csv_writer.writerow([
                episode,
                med_progress,
                win_rate,
                med_reward,
                round(agent.beta, 4),
                round(agent.learning_rate, 6),
                round(elapsed, 2)
            ])
            eval_csv_file.flush()

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.model.save(best_model_path)
                print(f"  New best model saved (win rate {win_rate:.2f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Best model saved to {best_model_path} (win rate {best_win_rate:.2f})")
    print(f"Training data saved to {csv_filename}")
    print(f"Evaluation snapshots saved to {eval_csv_filename}")

    csv_file.close()
    eval_csv_file.close()

if __name__ == "__main__":
    main()