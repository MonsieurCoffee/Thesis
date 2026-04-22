# analyze_scripts/exploration_quality.py
"""
Compare the informativeness of exploration: progress per click in early training.
"""

import os
import pandas as pd
import numpy as np

# ---------- Path Setup ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dqn_path = os.path.join(project_root, 'dqn', 'csv', 'training_dqn.csv')
cae_path = os.path.join(project_root, 'caedqn', 'csv', 'training_caedqn.csv')

# ---------- Load Data ----------
dqn = pd.read_csv(dqn_path)
cae = pd.read_csv(cae_path)

# Focus on early training: first 10,000 episodes
early_dqn = dqn[dqn['episode'] <= 10000]
early_cae = cae[cae['episode'] <= 10000]

# Compute progress per click for each episode
early_dqn['progress_per_click'] = early_dqn['progress'] / early_dqn['n_clicks']
early_cae['progress_per_click'] = early_cae['progress'] / early_cae['n_clicks']

# Aggregate statistics
mean_dqn = early_dqn['progress_per_click'].mean()
mean_cae = early_cae['progress_per_click'].mean()
median_dqn = early_dqn['progress_per_click'].median()
median_cae = early_cae['progress_per_click'].median()

print("=== Exploration Quality: Progress per Click (First 10,000 Episodes) ===")
print(f"DQN      : mean = {mean_dqn:.3f}, median = {median_dqn:.3f}")
print(f"CAE-DQN  : mean = {mean_cae:.3f}, median = {median_cae:.3f}")
print(f"Ratio CAE/DQN (mean): {mean_cae/mean_dqn:.2f}x")