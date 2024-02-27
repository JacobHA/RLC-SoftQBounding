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

def load_data(folder_name):
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
        # break

    # Take means and stds of the data:
    df = df.groupby('times').agg(['mean', 'std'])
    df = df.reset_index()

    return df

def add_to_plot(ax, df, name):
    ax.plot(df['times'], df[name]['mean'], name_to_color[name], label=name)
    if name in ['q', 'lb', 'ub']:
        ax.fill_between(df['times'], 
                        df[name]['mean'] - df[name+'std']['mean'], 
                        df[name]['mean'] + df[name+'std']['mean'], 
                        color=name_to_color[name], alpha=0.2)
    else:
        ax.fill_between(df['times'], 
                        df[name]['mean'] - df[name]['std'],
                        df[name]['mean'] + df[name]['std'],
                        color=name_to_color[name], alpha=0.2)

    
    return ax

# Get the data:
gamma=0.98
for data_folder in [f'clip-{gamma}data', f'{gamma}data', f'naiveclip-{gamma}data', f'lr0.00clip-{gamma}data']:
    df = load_data(data_folder)
    fig, ax = plt.subplots()
    
    plt.title(r'$Q$-values')
    for name in ['q', 'lb', 'ub']:
        ax = add_to_plot(ax, df, name)

    plt.legend()
    plt.savefig(f'{data_folder}/Q_values.png')
    plt.close()

    # Plot the rewards:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax = add_to_plot(ax, df, 'rewards')
    ax2 = add_to_plot(ax2, df, 'loss')
    ax2.set_yscale('log')
    plt.title('Training Metrics')
    plt.savefig(f'{data_folder}/rewards.png')
    plt.close()