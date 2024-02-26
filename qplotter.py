# Plot the data in the data folder, which contains Q, lb, ub, rewards, loss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gymnasium.wrappers import TimeLimit
from tabular import ModifiedFrozenLake, softq_solver

from visualization import plot_dist
name_to_color = {
    'q': 'k',
    'lb': 'b',
    'ub': 'r',
    'rewards': 'g',
    'loss': 'r'
}

FONTSIZE=18

SKIP_POINTS = 50

# First, let's solve the problem exactly, so we can put inset of the policy
# and value function:
env_str = '7x7zigzag'
env = ModifiedFrozenLake(map_name=env_str,cyclic_mode=False,slippery=0)
env = TimeLimit(env, max_episode_steps=1000)
# env = gymnasium.make('FrozenLake-v1', is_slippery=False)
beta = 5
gamma = 0.98
Q, V, pi = softq_solver(env, beta=beta, gamma=gamma, tolerance=1e-14)
# reshape V to be plottable:
V_re = V.flatten()#V.reshape(env.desc.shape)
plot_dist(env.desc, V_re, show_plot=False, filename='Vs.png', dpi=300)#, fontsize=FONTSIZE)


def load_data(folder):
    files = os.listdir(folder)
    # load in data from each file into a pandas dataframe:
    data = []
    # make a dataframe to load the files into:
    names = ['q', 'qstd', 'lb', 'lbstd', 'ub', 'ubstd', 'rewards', 'loss', 'times']
    df = pd.DataFrame(columns=names)
    for file in files:
        if file[-3:] != 'npy':
            continue
        path = os.path.join(folder, file)
        npz = np.load(path)
        new_df = pd.DataFrame(npz.T, columns=names)
        # add the new data to the df:
        df = pd.concat([df, new_df], axis=0)
        # break

    # Take means and stds of the data:
    df = df.groupby('times').agg(['mean', 'std'])
    df = df.reset_index()

    return df

def add_to_plot(ax, df, name, marker, label, color):
    smoothed_values = df[name]['mean'][::SKIP_POINTS]
    smoothed_stds = df[name+'std']['mean'][::SKIP_POINTS]
    # skip points across dataframe:
    ax.plot(df['times'][::SKIP_POINTS] / 10_000, smoothed_values, 
            label=f'{label}', linestyle='-', 
            marker=marker, markersize=8,
            color=color, lw=3)
    # add error bars:
    # ax.errorbar(df['times'][::SKIP_POINTS], smoothed_values, 
    #             yerr=df[name+'std']['mean'][::SKIP_POINTS], #/ np.sqrt(30),
    #             fmt='o', color=name_to_color[name])

    ax.fill_between(df['times'][::SKIP_POINTS] / 10_000, 
                    smoothed_values - smoothed_stds,
                    smoothed_values + smoothed_stds,
                    color=color, alpha=0.2)


    
    return ax

# Get the data:
gamma=0.98
fig, axes = plt.subplots(1,2, sharey=True, figsize=(12, 5))
markers = ['o', 's']
folders = [f'clip-{gamma}data', f'{gamma}data']
labels = ['Lower Bound','Q Values', 'Upper Bound']
colors = ['#0db8a1', '#000000', '#f01abe']
titles = ['Clipping During Training', 'No Clipping During Training']
for ax, folder, marker, title in zip(axes, folders, markers, titles):
    df = load_data(folder)
    
    for name, color, label in zip(['lb', 'q', 'ub'], colors, labels):
        ax = add_to_plot(ax, df, name, marker, label, color)
    # plt.legend()
    # ax.legend(loc='lower right', fontsize=FONTSIZE, frameon=True, fancybox=True, shadow=False)
    ax.set_xlabel(r'Environment Steps $\times 10^4$', fontsize=FONTSIZE)

    # Adjust legend placement
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3, fontsize=FONTSIZE)
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    # divide x axis by 10k:


    # Adjust tick marks and labels
    ax.tick_params(axis='both', which='both', labelsize=FONTSIZE)

    # Provide a light gray background
    ax.set_facecolor('#f5f5f5')
    ax.set_xlim(0, 8)#_000)
    ax.set_ylim(-100,20)
    # push the title down into the plot:
    # ax.set_title(title, fontsize=FONTSIZE, pad=-60, color='black')
    # Draw a white box with the title centered at the top of the plot
    box_props = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=18, color='black', ha='center', va='center', bbox=box_props)

    
# put a shared legend:
# make the legend:
labels=[ 'Lower Bound', 'Q Values', 'Upper Bound']
handles = [plt.Line2D([0], [0], color=colors[0], linestyle='-', lw=4),
           plt.Line2D([0], [0], color=colors[1], linestyle='-', lw=4),
           plt.Line2D([0], [0], color=colors[2], linestyle='-', lw=4)]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, shadow=False, ncol=3, fontsize=FONTSIZE)
# plt.title(r'Effect of Clipping on $Q$-values During Training')
# set the title for the entire plot:
# fig.suptitle(r'Effect of Bounds on $Q$-values', fontsize=22)

axes[0].set_ylabel(r'$Q$-Values', fontsize=16)
# reduce hspace between subplots:
# plt.subplots_adjust(hspace=0.1)

plt.rcParams.update({'figure.figsize': (60,60)})


plt.tight_layout()
plt.savefig(f'Q_values.png', bbox_inches='tight', dpi=600)
plt.close()

