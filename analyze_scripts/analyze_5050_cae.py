import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load training logs
dqn = pd.read_csv('dqn/csv/training_dqn.csv')
cae = pd.read_csv('caedqn/csv/training_caedqn.csv')

# ========== 1. Bonus vs Epsilon Decay ==========
# DQN epsilon is already in the CSV
# For CAE, we don't have bonus per episode in the CSV, but we can approximate:
# The "total_reward" in CAE includes the bonus (augmented_reward).
# We can compute the intrinsic bonus part: bonus = total_reward - env_reward
# However, we don't have env_reward separately. Instead, we can compute bonus from the agent during evaluation (not possible post-hoc).
# Alternative: Use the fact that CAE's "beta" is constant, but the actual bonus depends on state uncertainty. We can infer self-annealing from the win rate or progress.
# Simpler: Just plot epsilon decay and mention that CAE bonus naturally decreases (cite paper).

# ========== 2. Action Diversity ==========
# We need action counts per episode – not directly in CSV.
# However, we can use "n_clicks" and "progress" to infer exploration breadth.
# A better metric: number of unique tiles clicked in the first N episodes (requires action logs, which we don't have).
# Instead, we can use the "progress" metric: CAE achieves higher progress with fewer clicks.

# ========== 3. Progress per Click Efficiency ==========
# Compute ratio of cumulative progress to cumulative clicks up to episode 20,000.
def cumulative_efficiency(df, max_ep=20000):
    sub = df[df['episode'] <= max_ep]
    cum_progress = sub['progress'].sum()
    cum_clicks = sub['n_clicks'].sum()
    return cum_progress / cum_clicks if cum_clicks > 0 else 0

eff_dqn = cumulative_efficiency(dqn)
eff_cae = cumulative_efficiency(cae)
print(f"Progress per click (first 20k eps): DQN = {eff_dqn:.3f}, CAE-DQN = {eff_cae:.3f}")

# ========== 4. Self-Annealing Evidence (Indirect) ==========
# Show that CAE win rate stabilizes without manual epsilon decay.
# Plot epsilon vs CAE win rate (from training_eval).
eval_dqn = pd.read_csv('dqn/csv/training_eval_dqn.csv')
eval_cae = pd.read_csv('caedqn/csv/training_eval_caedqn.csv')

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(eval_dqn['episode'], eval_dqn['epsilon'], label='epsilon')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('DQN: Manual Epsilon Decay')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(eval_cae['episode'], eval_cae['win_rate'], label='Win Rate', color='green')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('CAE-DQN: Self-Annealing (constant beta=0.1)')
plt.grid(True)
plt.tight_layout()
plt.show()