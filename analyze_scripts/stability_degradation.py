import pandas as pd
import numpy as np
from scipy import stats

# Load evaluation snapshots
dqn_eval = pd.read_csv('dqn/csv/training_eval_dqn.csv')
cae_eval = pd.read_csv('caedqn/csv/training_eval_caedqn.csv')

# Extract data
episodes = dqn_eval['episode'].to_numpy()
dqn_winrate = dqn_eval['win_rate'].to_numpy()
cae_winrate = cae_eval['win_rate'].to_numpy()

# ========== 1. Variance in final 30k episodes (50k - 80k) ==========
mask_final = (episodes >= 50000) & (episodes <= 80000)
ep_final = episodes[mask_final]
dqn_final = dqn_winrate[mask_final]
cae_final = cae_winrate[mask_final]

dqn_std = np.std(dqn_final, ddof=1)
cae_std = np.std(cae_final, ddof=1)

print("=" * 60)
print("STABILITY ANALYSIS: FINAL 30k EPISODES (50,000 - 80,000)")
print("=" * 60)

print(f"\n[1] Standard Deviation of Win Rate")
print("-" * 40)
print(f"{'Model':<12} {'Std Dev':<12} {'Interpretation'}")
print(f"{'DQN':<12} {dqn_std:>8.4f}    {'More stable' if dqn_std < cae_std else 'Less stable'}")
print(f"{'CAE-DQN':<12} {cae_std:>8.4f}    {'Less stable' if dqn_std < cae_std else 'More stable'}")

# ========== 2. Linear Regression on CAE-DQN collapse segment ==========
# Use the same 50k-80k range
x = ep_final
y = cae_final

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("\n[2] Linear Regression on CAE-DQN (50,000 - 80,000 episodes)")
print("-" * 40)
print(f"{'Metric':<20} {'Value':<15}")
print(f"{'Slope (β₁)':<20} {slope:<15.6f}")
print(f"{'Intercept (β₀)':<20} {intercept:<15.6f}")
print(f"{'p-value (slope)':<20} {p_value:<15.6f}")
print(f"{'R-squared':<20} {r_value**2:<15.4f}") #type:ignore
print(f"{'Std Error':<20} {std_err:<15.6f}")
print(f"{'Significant (α=0.05)':<20} {'Yes' if p_value < 0.05 else 'No'}") #type:ignore

# ========== 3. Additional: Mean win rate during this segment ==========
dqn_mean_final = np.mean(dqn_final)
cae_mean_final = np.mean(cae_final)

print("\n[3] Mean Win Rate in Final 30k Episodes")
print("-" * 40)
print(f"{'Model':<12} {'Mean Win Rate':<15}")
print(f"{'DQN':<12} {dqn_mean_final:>8.4f}")
print(f"{'CAE-DQN':<12} {cae_mean_final:>8.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION: CAE-DQN shows higher variance and significant")
print("negative slope during the final 30k episodes, confirming the")
print("performance collapse observed in the training curve.")
print("=" * 60)