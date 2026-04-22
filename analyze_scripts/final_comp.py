import numpy as np
from scipy import stats

# Data: win rates per seed (5 seeds each)
dqn_winrates = np.array([0.590, 0.590, 0.566, 0.582, 0.572])
cae_winrates = np.array([0.726, 0.722, 0.724, 0.746, 0.712])

# Descriptive statistics
mean_dqn = np.mean(dqn_winrates)
std_dqn = np.std(dqn_winrates, ddof=1)
mean_cae = np.mean(cae_winrates)
std_cae = np.std(cae_winrates, ddof=1)

# Shapiro-Wilk test
w_dqn, p_shapiro_dqn = stats.shapiro(dqn_winrates)
w_cae, p_shapiro_cae = stats.shapiro(cae_winrates)

# Independent t-test (Welch's)
t_stat, p_value = stats.ttest_ind(cae_winrates, dqn_winrates, equal_var=False)

# Degrees of freedom for Welch's t-test
n1, n2 = len(cae_winrates), len(dqn_winrates)
var1, var2 = np.var(cae_winrates, ddof=1), np.var(dqn_winrates, ddof=1)
se1, se2 = var1/n1, var2/n2
df_welch = (se1 + se2)**2 / (se1**2/(n1-1) + se2**2/(n2-1))

# Cohen's d
pooled_std = np.sqrt((std_dqn**2 + std_cae**2) / 2)
cohen_d = (mean_cae - mean_dqn) / pooled_std

# 95% CI parametric (using t-distribution with Welch df)
diff = mean_cae - mean_dqn
se = np.sqrt(var1/n1 + var2/n2)
t_crit = stats.t.ppf(0.975, df_welch)
ci_lower = diff - t_crit * se
ci_upper = diff + t_crit * se

# Bootstrap 95% CI
np.random.seed(42)
n_bootstrap = 10000
boot_diffs = []
for _ in range(n_bootstrap):
    dqn_sample = np.random.choice(dqn_winrates, size=n2, replace=True)
    cae_sample = np.random.choice(cae_winrates, size=n1, replace=True)
    boot_diffs.append(np.mean(cae_sample) - np.mean(dqn_sample))
ci_boot_lower = np.percentile(boot_diffs, 2.5)
ci_boot_upper = np.percentile(boot_diffs, 97.5)

# Pretty table output
print("\n" + "="*70)
print("STATISTICAL ANALYSIS: FINAL PERFORMANCE COMPARISON")
print("="*70)

print("\n[1] DESCRIPTIVE STATISTICS")
print("-" * 50)
print(f"{'Group':<15} {'N':<5} {'Mean (%)':<12} {'SD (%)':<10}")
print(f"{'DQN':<15} {n2:<5} {mean_dqn*100:>6.1f}      {std_dqn*100:>6.2f}")
print(f"{'CAE-DQN':<15} {n1:<5} {mean_cae*100:>6.1f}      {std_cae*100:>6.2f}")

print("\n[2] NORMALITY TEST (Shapiro-Wilk)")
print("-" * 50)
print(f"{'Group':<15} {'W statistic':<12} {'p-value':<10} {'Normal?'}")
print(f"{'DQN':<15} {w_dqn:<12.4f} {p_shapiro_dqn:<10.4f} {'Yes' if p_shapiro_dqn > 0.05 else 'No'}")
print(f"{'CAE-DQN':<15} {w_cae:<12.4f} {p_shapiro_cae:<10.4f} {'Yes' if p_shapiro_cae > 0.05 else 'No'}")

print("\n[3] INDEPENDENT T-TEST (Welch's)")
print("-" * 50)
print(f"{'Test':<25} {'Value':<15}")
print(f"{'t-statistic':<25} {t_stat:<15.4f}")
print(f"{'Degrees of freedom':<25} {df_welch:<15.2f}")
print(f"{'p-value':<25} {p_value:<15.6f}")
print(f"{'Significant (α=0.05)':<25} {'Yes' if p_value < 0.05 else 'No'}") #type:ignore

print("\n[4] EFFECT SIZE")
print("-" * 50)
print(f"{'Metric':<25} {'Value':<15} {'Interpretation'}")
print(f"{'Cohen''s d':<25} {cohen_d:<15.3f} {'Very large (>0.8)' if cohen_d > 0.8 else 'Large' if cohen_d > 0.5 else 'Medium' if cohen_d > 0.2 else 'Small'}")

print("\n[5] CONFIDENCE INTERVALS (95%)")
print("-" * 50)
print(f"{'Method':<25} {'Lower':<12} {'Upper':<12}")
print(f"{'Parametric (Welch df)':<25} {ci_lower*100:>6.2f}%     {ci_upper*100:>6.2f}%")
print(f"{'Bootstrap (10k resamples)':<25} {ci_boot_lower*100:>6.2f}%     {ci_boot_upper*100:>6.2f}%")

print("\n" + "="*70)
print("INTERPRETATION: CAE-DQN significantly outperforms DQN.")
print("="*70 + "\n")