import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from avg_rewards.csv:
df = pd.read_csv('robust_avg_rewards.csv')

# Define a function to calculate SEM
def sem(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

# Get the number of unique learning rates:
lrs = df['lr'].unique()

# Aggregate by group, calculating mean and SEM
result = df.groupby(['lr', 'clip', 'naive']).agg(['mean', sem, 'count'])

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


labels = ['Clipping:\nProposed Bounds', "Clipping:\nBaseline Bounds", 'No Clipping']
colors = ['#FF5733', '#007acc',  '#333333']
markers = ['o', '^', 's']
def plot_all(value):
    # Plot each of the methods in a different color:
    for method, label, color, marker in zip(methods.keys(), labels, colors, markers):
        clip = methods[method]["clip"]
        naive = methods[method]["naive"]

        # get rows with clip == clip and naive == naive:
        subdf = df[(df['clip'] == clip) & (df['naive'] == naive)]
        # Look at the lr and value columns:
        subdf = subdf[['lr', value]]
        if not subdf.empty:
            norm = 1# len(subdf)**0.5
            # norm = 10**0.5
            WINDOW = 1
            # Smooth out the plot using a moving average with increased window_length
            rwds = subdf[value]['mean']
            rwds = rwds.rolling(window=WINDOW).mean()
            # Also smooth the standard deviation:
            std = subdf[value]['sem']
            std = std.rolling(window=WINDOW).mean()
            if method == 'naive':
                markersize = 4
            else:
                markersize = 4
            plt.plot(subdf['lr'], rwds, label=label, color=color, marker=marker, markersize=markersize)
            plt.fill_between(subdf['lr'], rwds - std / norm,
                            rwds + std / norm, alpha=0.2,
                                color=color)

    plt.xlabel('Learning Rate')
    if value == 'avg_reward':
        plt.ylabel('Total Integrated Evaluation Reward (AUC)')
    elif value == 'avg_gap':
        plt.ylabel('Average Gap Between Lower and Upper Bounds')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=False, ncol=3, fontsize=12)
    # use grid lines with sns style:
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlim(0,0.0001)

    #plt.title('Learning Rate Sensitivity')

    plt.tight_layout()
    plt.savefig(f'visualizations/lr_sensitivity{value}.png', dpi=300)
    plt.close()

plot_all('avg_reward')
plot_all('avg_gap')