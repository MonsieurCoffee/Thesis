# simple_eval.py
import numpy as np
import tensorflow as tf
from minesweeper_env import MinesweeperEnv

# ==================== CONFIGURATION ====================
DQN_MODEL_PATH = 'dqn/dqn_best.keras'
CAEDQN_MODEL_PATH = 'caedqn/caedqn_best.keras'
EVAL_SEED = 300
EPISODES = 50          # change this if you want more/less episodes
ROWS, COLS, MINES = 6, 6, 4
# =======================================================

def evaluate(model_path, agent_name):
    print(f"\n--- Evaluating {agent_name} on seed {EVAL_SEED} for {EPISODES} episodes ---")
    model = tf.keras.models.load_model(model_path)
    env = MinesweeperEnv(width=ROWS, height=COLS, n_mines=MINES, seed=EVAL_SEED)
    
    wins = 0
    for ep in range(1, EPISODES + 1):
        env.reset()
        done = False
        while not done:
            state = env.state_im
            unknown = [i for i, t in enumerate(env.state) if t['value'] == 'U']
            if not unknown:
                break
            
            state_batch = np.expand_dims(state, axis=0)
            outputs = model.predict(state_batch, verbose=0)
            
            # Handle both DQN (single output) and CAE-DQN (list of outputs)
            if isinstance(outputs, (list, tuple)):
                q_values = outputs[0][0]      # CAE-DQN
            else:
                q_values = outputs[0]         # DQN
            
            # Mask already revealed tiles
            mask = np.ones(env.n_tiles, dtype=bool)
            mask[unknown] = False
            q_values[mask] = -np.inf
            action = int(np.argmax(q_values))
            
            _, reward, done = env.step(action)
        
        if env.n_wins > 0:
            wins += 1
        
        # Simple progress print every 100 episodes
        if ep % 100 == 0:
            print(f"  Episode {ep}/{EPISODES} | Current win rate: {wins/ep:.2%}")
    
    final_rate = wins / EPISODES
    print(f"\n>>> {agent_name} final win rate on seed {EVAL_SEED}: {final_rate:.2%} ({wins}/{EPISODES})")
    return final_rate

if __name__ == "__main__":
    print("Loading models and starting evaluation...")
    dqn_rate = evaluate(DQN_MODEL_PATH, "DQN")
    cae_rate = evaluate(CAEDQN_MODEL_PATH, "CAE-DQN")
    
    print("\n" + "="*50)
    print(f"SUMMARY (seed={EVAL_SEED}, episodes={EPISODES})")
    print(f"DQN     : {dqn_rate:.2%}")
    print(f"CAE-DQN : {cae_rate:.2%}")
    print("="*50)