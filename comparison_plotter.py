import os
from matplotlib.backend_bases import FigureCanvasBase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter
import matplotlib.image as mpimg 
from visualization import plot_dist

name_to_color = {
    'q': 'k',
    'lb': 'b',
    'ub': 'r',
    'rewards': 'g',
    'loss': 'r'
}

FONTSIZE = 18
from gymnasium.wrappers import TimeLimit
from tabular import ModifiedFrozenLake
SKIP_POINTS = 10

# First, let's solve the problem exactly, so we can put inset of the policy
# and value function:
env_str = '7x7zigzag'
env = ModifiedFrozenLake(map_name=env_str,cyclic_mode=False,slippery=0)
env = TimeLimit(env, max_episode_steps=1000)
# env = gymnasium.make('FrozenLake-v1', is_slippery=False)
beta = 5
gamma = 0.98
from tabular import ModifiedFrozenLake, softq_solver
Q, V, pi = softq_solver(env, beta=beta, gamma=gamma, tolerance=1e-14)
plot_dist(env.desc, pi, show_plot=False, filename='policy.png', dpi=600)#, fontsize=FONTSIZE)

def load_data(data_folder):
    files = os.listdir(data_folder)
    # load in data from each file into a pandas dataframe:
    data = []
    # make a dataframe to load the files into:
    names = ['q', 'qstd', 'lb', 'lbstd', 'ub', 'ubstd', 'rewards', 'loss', 'times']
    df = pd.DataFrame(columns=names)
    for file in files:
        if file[-3:] != 'npy':
            continue
        path = os.path.join(data_folder, file)
        npz = np.load(path)
        # Normalize the rewards between 0 and 1:
        npz[-3,:] = (npz[-3,:] - npz[-3,:].min()) / (-508 - -1000)#(npz[-3,:].max() - npz[-3,:].min())
        new_df = pd.DataFrame(npz.T, columns=names)
        # add the new data to the df:
        df = pd.concat([df, new_df], axis=0)

    # Take means and standard error of the mean
    df = df.groupby('times').agg(['mean', 'std'])
    df = df.reset_index()

    return df


def add_to_plot(ax, df, name, data_label, color, marker):
    # Smooth out the plot using a moving average with increased window_length
    smoothed_values = df[name]['mean'][::SKIP_POINTS]
    smoothed_values = savgol_filter(df[name]['mean'], window_length=10, polyorder=3)[::SKIP_POINTS]

    ax.plot(df['times'][::SKIP_POINTS] / 10_000, smoothed_values, label=f'{data_label}', 
            color=color, linestyle='-', marker=marker, markersize=8, lw=3)
    
    n_runs = 30#len(df[name]['mean'])
    if name in ['q', 'lb', 'ub']:
        ax.fill_between(df['times'][::SKIP_POINTS] / 10_000, 
                        smoothed_values - df[name+'std']['mean'][::SKIP_POINTS] / np.sqrt(n_runs),
                        smoothed_values + df[name+'std']['mean'][::SKIP_POINTS] / np.sqrt(n_runs),
                        alpha=0.2, color=color)
    else:
        ax.fill_between(df['times'][::SKIP_POINTS] / 10_000, 
                        smoothed_values - df[name]['std'][::SKIP_POINTS] / np.sqrt(n_runs),
                        smoothed_values + df[name]['std'][::SKIP_POINTS] / np.sqrt(n_runs),
                        alpha=0.2, color=color)

    return ax

# sns.set_style('whitegrid')
plt.rcParams.update({'font.size': FONTSIZE, 'axes.labelcolor': 'black'})
# families =
plt.rc('font', family='sans-serif', serif='Helvetica')

# Set a fixed width for the figure to prevent horizontal shrinking
fig, ax = plt.subplots(figsize=(10, 6))

folders = ['clip-0.98data', 'naiveclip-0.98data', '0.98data']
labels = ['Clipping:\nProposed Bounds', "Clipping:\nBaseline Bounds", 'No Clipping']
colors = ['#FF5733', '#007acc',  '#333333']
markers = ['o', '^', 's']

for label, folder, color, marker in zip(labels, folders, colors, markers):
    df = load_data(folder)
    ax = add_to_plot(ax, df, 'rewards', label, color, marker)

# plt.title(r'Effect of Bounds in Soft $Q$-Learning ', fontsize=1.5*FONTSIZE, pad=20)  # Add padding to the title
plt.xlabel(r'Environment Steps $\times 10^4$', fontsize=FONTSIZE)
plt.ylabel('Evaluation Reward', fontsize=FONTSIZE)
plt.ylim(-0.01, 1.01)
plt.xlim(0, 8)#_000)

# Adjust legend placement
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3, fontsize=FONTSIZE)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust tick marks and labels
ax.tick_params(axis='both', which='both', labelsize=FONTSIZE)

# Provide a light gray background
ax.set_facecolor('#f5f5f5')

plt.rcParams.update({'figure.figsize': (60,60)})

plt.tight_layout()

# Add the inset plot using the saved image
inset = fig.add_axes([0.557, 0.4, 0.4, 0.4])

# Load the saved image
inset_image_path = 'policy.png'
inset_image = mpimg.imread(inset_image_path, format='png')

# Display the image in the inset with high resolution:
inset.imshow(inset_image)#, interpolation='none')
inset.axis('off')  

# Save the total figure
fig.savefig('rewards_comparison.png', bbox_inches='tight', dpi=600)