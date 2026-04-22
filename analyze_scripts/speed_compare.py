import pandas as pd
import matplotlib.pyplot as plt

# Load training evaluation logs
df_dqn = pd.read_csv("dqn/csv/training_eval_dqn.csv")
df_cae = pd.read_csv("caedqn/csv/training_eval_caedqn.csv")

# Find peak win rates and their corresponding times
peak_dqn = df_dqn.loc[df_dqn['win_rate'].idxmax()]
peak_cae = df_cae.loc[df_cae['win_rate'].idxmax()]

plt.figure(figsize=(12, 6))

# Plot win rate curves with markers
plt.plot(df_dqn['time_elapsed'] / 60, df_dqn['win_rate'],
         marker='o', markersize=4, linewidth=2, alpha=0.8,
         label='DQN (ε-greedy)')
plt.plot(df_cae['time_elapsed'] / 60, df_cae['win_rate'],
         marker='s', markersize=4, linewidth=2, alpha=0.8,
         label='CAE-DQN (Critic as Explorer)')

# Vertical lines for peak times
plt.axvline(x=peak_dqn['time_elapsed'] / 60, color='blue', linestyle='--', alpha=0.5) #type:ignore
plt.axvline(x=peak_cae['time_elapsed'] / 60, color='red', linestyle='--', alpha=0.5) #type:ignore

# Annotate peak times
plt.text(peak_dqn['time_elapsed'] / 75 - 2, peak_dqn['win_rate'] + 0, #type:ignore
         f"DQN peak: {peak_dqn['win_rate']:.2f} at {peak_dqn['time_elapsed']/60:.1f} min",
         color='black', fontsize=10)
plt.text(peak_cae['time_elapsed'] / 60 + 2, peak_cae['win_rate'] - 0, #type:ignore
         f"CAE peak: {peak_cae['win_rate']:.2f} at {peak_cae['time_elapsed']/60:.1f} min",
         color='black', fontsize=10)

plt.xlabel('Training Time (minutes)')
plt.ylabel('Win Rate (over last 1000 episodes)')
plt.title('Minesweeper 6x6 with 4 Mines - Training Win Rate vs. Training Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print exact values for reference
print(f"DQN peak win rate: {peak_dqn['win_rate']:.3f} at {peak_dqn['time_elapsed']/60:.1f} minutes")
print(f"CAE-DQN peak win rate: {peak_cae['win_rate']:.3f} at {peak_cae['time_elapsed']/60:.1f} minutes")