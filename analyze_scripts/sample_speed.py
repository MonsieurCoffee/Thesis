import pandas as pd
import numpy as np
from scipy import integrate
import pymannkendall as mk

# Load evaluation snapshots (every 1000 episodes)
dqn_eval = pd.read_csv('dqn/csv/training_eval_dqn.csv')
cae_eval = pd.read_csv('caedqn/csv/training_eval_caedqn.csv')

# Extract win rate series
dqn_winrate = dqn_eval['win_rate'].values
cae_winrate = cae_eval['win_rate'].values
episodes = dqn_eval['episode'].values  # same for both

# Total number of evaluation points
n_points = len(episodes)

# ========== 1. Area Under the Curve (first 40k episodes) ==========
# Find index up to 40,000 episodes
idx_40k = np.searchsorted(episodes, 40000, side='right') #type:ignore
if idx_40k == 0:
    idx_40k = n_points  # fallback if no data before 40k

ep_40k = episodes[:idx_40k]
dqn_win_40k = dqn_winrate[:idx_40k]
cae_win_40k = cae_winrate[:idx_40k]

# Trapezoidal integration
dqn_auc = integrate.trapezoid(dqn_win_40k, ep_40k)
cae_auc = integrate.trapezoid(cae_win_40k, ep_40k)

print("=" * 60)
print("SAMPLE EFFICIENCY ANALYSIS")
print("=" * 60)

print(f"\n[1] AUC (First {ep_40k[-1]} episodes)")
print("-" * 40)
print(f"{'Model':<12} {'AUC':<12} {'Relative to DQN'}")
print(f"{'DQN':<12} {dqn_auc:>8.1f}    {'1.00x'}")
print(f"{'CAE-DQN':<12} {cae_auc:>8.1f}    {cae_auc/dqn_auc:>5.2f}x")

# ========== 2. Episode to reach 50% win rate ==========
def first_episode_threshold(winrate_series, episodes, threshold=0.5):
    """Return first episode where win rate >= threshold, or None if never reached."""
    for wr, ep in zip(winrate_series, episodes):
        if wr >= threshold:
            return ep
    return None

dqn_ep50 = first_episode_threshold(dqn_winrate, episodes, 0.5)
cae_ep50 = first_episode_threshold(cae_winrate, episodes, 0.5)

print("\n[2] Episode to reach 50% win rate")
print("-" * 40)
print(f"{'Model':<12} {'First Episode ≥50%'}")
print(f"{'DQN':<12} {dqn_ep50 if dqn_ep50 else 'Never reached'}")
print(f"{'CAE-DQN':<12} {cae_ep50 if cae_ep50 else 'Never reached'}")

# ========== 3. Mann-Kendall Trend Test ==========
# For DQN (full training)
mk_dqn = mk.original_test(dqn_winrate)
mk_cae = mk.original_test(cae_winrate)

print("\n[3] Mann-Kendall Trend Test (Full Training)")
print("-" * 40)
print(f"{'Model':<12} {'Trend':<12} {'p-value':<10} {'Slope (Sen)':<12}")
print(f"{'DQN':<12} {mk_dqn.trend:<12} {mk_dqn.p:<10.4f} {mk_dqn.slope:<12.6f}")
print(f"{'CAE-DQN':<12} {mk_cae.trend:<12} {mk_cae.p:<10.4f} {mk_cae.slope:<12.6f}")

print("\n" + "=" * 60)