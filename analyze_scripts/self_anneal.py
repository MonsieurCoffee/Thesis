# analyze_scripts/self_anneal.py
"""
Plot the self‑annealing behaviour of the CAE exploration bonus.
Displays the smoothed curve on screen (no file saved).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Path Setup ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)                     # experiment/
csv_path = os.path.join(project_root, 'caedqn', 'csv', 'training_caedqn_anneal.csv')

# ---------- Load Data ----------
df = pd.read_csv(csv_path)

# Simple Moving Average with a window of 500 episodes
window = 500
df['bonus_sma'] = df['avg_bonus'].rolling(window=window, min_periods=1).mean()

# ---------- Plot ----------
plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['bonus_sma'], color='crimson', linewidth=1.8,
         label=f'SMA {window} episodes')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average CAE Exploration Bonus', fontsize=12)
plt.title('Self‑Annealing of CAE Bonus During Training', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Show the plot (no saving)
plt.show()