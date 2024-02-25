# Plot the data in the data folder, which contains Q, lb, ub, rewards, loss
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

name_to_color = {
    'q': 'k',
    'lb': 'b',
    'ub': 'r',
    'rewards': 'g',
    'loss': 'r'
}
SKIP_POINTS = 20

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

def add_to_plot(ax, df, name, marker, label):
    smoothed_values = df[name]['mean'][::SKIP_POINTS]
    # skip points across dataframe:
    ax.plot(df['times'][::SKIP_POINTS], smoothed_values, 
            label=f'{label}: {name.upper()}', linestyle='-', 
            marker=marker, markersize=8,
            color=name_to_color[name])
    # add error bars:
    ax.errorbar(df['times'][::SKIP_POINTS], smoothed_values, 
                yerr=df[name+'std']['mean'][::SKIP_POINTS], #/ np.sqrt(30),
                fmt='o', color=name_to_color[name])

        # ax.fill_between(df['times'], 
        #                 df[name]['mean'] - df[name+'std']['mean'], 
        #                 df[name]['mean'] + df[name+'std']['mean'], 
        #                 color=name_to_color[name], alpha=0.2)


    
    return ax

# Get the data:
gamma=0.98
fig, axes = plt.subplots(1,2, sharey=True, figsize=(12, 5))
markers = ['o', 's']
folders = [f'clip-{gamma}data', f'{gamma}data']
labels = ['Clip', 'No Clip']
for ax, folder, marker, label in zip(axes, folders, markers, labels):
    df = load_data(folder)
    
    for name in ['q', 'lb', 'ub']:
        ax = add_to_plot(ax, df, name, marker, label)
    # plt.legend()
    ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=False)
    ax.set_xlabel('Environment Steps', fontsize=16)

# plt.title(r'Effect of Clipping on $Q$-values During Training')
# set the title for the entire plot:
fig.suptitle(r'Effect of Clipping on $Q$-values During Training', fontsize=22)

axes[0].set_ylabel(r'$Q$-Values', fontsize=16)
# reduce hspace between subplots:
plt.subplots_adjust(hspace=0.1)
plt.savefig(f'Q_values.png')
plt.close()

