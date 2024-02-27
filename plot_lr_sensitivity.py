import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from avg_rewards.csv:
df = pd.read_csv('avg_rewards.csv')

# Define a function to calculate SEM
def sem(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

# Aggregate by group, calculating mean and SEM
result = df.groupby(['lr', 'clip', 'naive'])['avg_reward'].agg(['mean', sem])

# Rename columns for clarity
# result.columns = ['Mean', 'SEM']

# df = df.groupby(['lr', 'clip', 'naive']).agg({'avg_reward': ['mean', 'std']})
df = result.reset_index()

# Define a mapping for methods
methods = {
    'hard': {'clip': True, 'naive': False},
    'naive': {'clip': True, 'naive': True},
    'none': {'clip': False, 'naive': False},
}

# Use map to look up the method based on 'clip' and 'naive' values
# df['method'] = df.apply(lambda row: next((key for key, value in methods.items() if value == {'clip': row['clip'], 'naive': row['naive']}), ''), axis=1)

# Drop unnecessary columns
# df = df.drop(columns=[('avg_reward', 'mean'), ('avg_reward', 'std')])

# Reorder columns for clarity (optional)
# df = df[['lr', 'clip', 'naive', 'method', 'avg_reward_mean', 'avg_reward_std']]

# Rename columns for clarity
# df.columns = ['lr', 'clip', 'naive', 'method', 'avg_reward_mean', 'avg_reward_std']


labels = ['Clipping:\nProposed Bounds', "Clipping:\nBaseline Bounds", 'No Clipping']
colors = ['#FF5733', '#007acc',  '#333333']
markers = ['o', '^', 's']

# Plot each of the methods in a different color:
for method, label, color, marker in zip(methods.keys(), labels, colors, markers):
    clip = methods[method]["clip"]
    naive = methods[method]["naive"]

    # get rows with clip == clip and naive == naive:
    subdf = df[(df['clip'] == clip) & (df['naive'] == naive)]#[1:]
    if not subdf.empty:
        norm = 1# len(subdf)**0.5
        # norm = 10**0.5
        WINDOW = 1
        # Smooth out the plot using a moving average with increased window_length
        rwds = subdf['mean']
        rwds = rwds.rolling(window=WINDOW).mean()
        # Also smooth the standard deviation:
        std = subdf['sem']
        std = std.rolling(window=WINDOW).mean()
        if method == 'naive':
            markersize = 8
        else:
            markersize = 4
        plt.plot(subdf['lr'], rwds, label=label, color=color, marker=marker, markersize=markersize)
        plt.fill_between(subdf['lr'], rwds - std / norm,
                         rwds + std / norm, alpha=0.2,
                            color=color)

plt.xlabel('Learning Rate')
plt.ylabel('Total Integrated Evaluation Reward (AUC)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=False, ncol=3, fontsize=12)
# use grid lines with sns style:
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0,0.0001)

#plt.title('Learning Rate Sensitivity')

plt.tight_layout()
plt.savefig('visualizations/lr_sensitivity.png', dpi=300)
