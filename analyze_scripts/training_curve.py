import pandas as pd
import matplotlib.pyplot as plt

# Paths to your two different CSV files
CSV_MODEL_1 = "dqn/csv/training_dqn.csv"
CSV_MODEL_2 = "caedqn/csv/training_caedqn.csv"

WINDOW_SIZE = 500 

def plot_comparison_window(df1, df2, col_name, display_name, color):
    # Change to 2 rows, 1 column. Adjust figsize to be taller (e.g., 8x10)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f"{display_name}", fontsize=16, y=0.93)

    # Plot DQN (Top - Index 0)
    plot_sub(axes[0], df1, col_name, f"DQN", color, True)
    
    # Plot CAE-DQN (Bottom - Index 1)
    plot_sub(axes[1], df2, col_name, f"CAE-DQN", color, True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type:ignore

def plot_sub(ax, df, column, title, color, smooth):
    if column not in df.columns:
        ax.set_title(f"{column} not found")
        return

    if smooth:
        # Plot raw data in faint gray
        ax.plot(df["episode"], df[column], color='gray', alpha=0.15, linewidth=0.5)
        # Plot rolling mean trend
        smooth_y = df[column].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        ax.plot(df["episode"], smooth_y, color=color, linewidth=2)
    else:
        ax.plot(df["episode"], df[column], color=color, linewidth=1.5)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.2)

def main():
    try:
        df1 = pd.read_csv(CSV_MODEL_1)
        df2 = pd.read_csv(CSV_MODEL_2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # List of metrics to generate windows for
    metrics = [
        ("total_reward", "Total Reward", "tab:blue"),
        #("win", "Win Rate", "green"),
        ("n_clicks", "Clicks", "red"),
    ]

    # Create an independent window for each metric
    for col_name, display_name, color in metrics:
        plot_comparison_window(df1, df2, col_name, display_name, color)

    # Open all windows
    plt.show()

if __name__ == "__main__":
    main()