import pandas as pd

# 1. Load both datasets
df_dqn = pd.read_csv('dqn/csv/eval_dqn_4mines.csv')
df_cae = pd.read_csv('caedqn/csv/eval_caedqn_4mines.csv')

def get_summary(df, name):
    # Group by seed and calculate mean
    summary = df.groupby('eval_seed')[['win', 'progress', 'total_reward', 'n_clicks']].mean()
    
    # Add Overall Average
    summary.loc['Overall'] = summary.mean()
    
    # Add a column to identify the model
    summary.insert(0, 'Model', name)
    return summary

# 2. Get summaries for both
summary_dqn = get_summary(df_dqn, 'DQN')
summary_cae = get_summary(df_cae, 'CAE-DQN')

# 3. Combine them into one "Master Table"
comparison_table = pd.concat([summary_dqn, summary_cae])

# 4. Clean up formatting
comparison_table.index.name = 'Seed'
print(comparison_table.round(3))

# Optional: Save to a CSV so you can open it in Excel
# comparison_table.to_csv('model_comparison.csv')