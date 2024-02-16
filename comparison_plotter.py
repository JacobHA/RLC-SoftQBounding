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
        new_df = pd.DataFrame(npz.T, columns=names)
        # add the new data to the df:
        df = pd.concat([df, new_df], axis=0)

    # Take means and stds of the data:
    df = df.groupby('times').agg(['mean', 'std'])
    df = df.reset_index()

    return df

def add_to_plot(ax, df, name, data_label):
    ax.plot(df['times'], df[name]['mean'], label=f'{data_label}')
    if name in ['q', 'lb', 'ub']:
        ax.fill_between(df['times'], 
                        df[name]['mean'] - df[name+'std']['mean'], 
                        df[name]['mean'] + df[name+'std']['mean'], 
                        alpha=0.2)
    else:
        ax.fill_between(df['times'], 
                        df[name]['mean'] - df[name]['std'],
                        df[name]['mean'] + df[name]['std'],
                        alpha=0.2)

    return ax

# Plot the rewards:
fig, ax = plt.subplots()

# Get the data:
for data_folder, data_label in [('clip-0.98data', 'Clipped'), ('0.98data', 'Unclipped')]:
    df = load_data(data_folder)

    ax = add_to_plot(ax, df, 'rewards', data_label)

plt.title('Evaluation Reward Comparison')
plt.xlabel('Environment Steps')
plt.ylabel('Reward')
plt.xlim(0,5e5)
plt.xticks([0, 1e5, 2e5, 3e5, 4e5, 5e5])
plt.legend()
plt.savefig('rewards_comparison.png')
# plt.show()
