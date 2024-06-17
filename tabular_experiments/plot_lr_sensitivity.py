import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define a mapping for methods
methods = {
    'hard': {'clip': True, 'naive': False},
    'naive': {'clip': True, 'naive': True},
    'none': {'clip': False, 'naive': False},
}

def auc_to_step(auc, max_step=200_000):
    # Convert AUC to step function:
    return max_step*(1 - auc)

def preprocess(csv_file, steps=200_000):
    # Load the data from avg_rewards.csv:
    df = pd.read_csv(csv_file)

    # Define a function to calculate SEM
    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    # apply the func auc_to_step to the [value]:
    df['avg_reward'] = df['avg_reward']#.apply(auc_to_step, args=(steps,))

    # Aggregate by group, calculating mean and SEM
    result = df.groupby(['lr', 'clip', 'naive']).agg(['mean', sem, 'count'])

    # Rename columns for clarity
    # result.columns = ['Mean', 'SEM']

    # df = df.groupby(['lr', 'clip', 'naive']).agg({'avg_reward': ['mean', 'std']})
    df = result.reset_index()
    return df


def plot_all(ax, csv_file, value, 
             custom_names=None, 
             custom_symbols=None,
             custom_colors=None,
             name=None
             ):

    if csv_file == 'lr_sweep_propalgo_fast.csv':
        num_steps = 200
    else:
        num_steps = 600_000
    df = preprocess(csv_file, steps=num_steps)

    # Plot each of the methods in a different color:
    if custom_names is not None:
        # assert len(custom_names) == len(), "Custom names must have the same length as the methods dict."
        labels = custom_names
    else:
        labels = ['Clipping:\nProposed\nBounds', "Clipping:\nBaseline Bounds", 'No Clipping']

    if custom_symbols is not None:
        markers = custom_symbols
    else:
        markers = ['o', '^', 's']

    if custom_colors is not None:
        colors = custom_colors
    else:
        colors = ['#FF5733', '#007acc',  '#333333']

    # if csv_file == 'lr_sweep.csv':
    #     # Remove the hard clip method from plotting:
    #     labels = labels[1:]
    #     colors = colors[1:]
    #     markers = markers[1:]

    for method, label, color, marker in zip(methods.keys(), labels, colors, markers):
        clip = methods[method]["clip"]
        naive = methods[method]["naive"]
        if csv_file == 'lr_sweep.csv':
            if method == 'hard':
                continue

        # get rows with clip == clip and naive == naive:
        subdf = df[(df['clip'] == clip) & (df['naive'] == naive)]
        # Look at the lr and value columns:
        subdf = subdf[['lr', value]]
        if not subdf.empty:
            WINDOW = 1
            # Smooth out the plot using a moving average with increased window_length
            rwds = subdf[value]['mean']
            rwds = rwds.rolling(window=WINDOW).mean()
            if value == 'avg_gap':
                rwds = np.abs(rwds)
            # Also smooth the standard deviation:
            std = subdf[value]['sem']
            std = std.rolling(window=WINDOW).mean()
            markersize = 10
            if marker == '*':
                markersize = 16
            linewidth = 3
            if value == 'avg_gap':
                # Rescale the x axis by a factor of lr:
                x = subdf['lr']
                # Add a bold line at y=0 to indicate optimal gap / tight bound:
                ax.axhline(0, color='k', linestyle='--', alpha=0.5)

            elif value == 'avg_reward':
                x = subdf['lr']#np.abs(df[(df['clip'] == clip) & (df['naive'] == naive)]['avg_gap']['mean'])/subdf['lr']
                # Rescale the x axis by a factor of lr:
                # x = subdf['lr']
                # markersize=4

            # ax.plot(subdf['lr'], rwds, label=label, color=color, marker=marker, markersize=markersize)
            # plt.fill_between(subdf['lr'], rwds - std,
            #                 rwds + std, alpha=0.2,
            #                     color=color)
            # Plot with shading for error:
            ax.plot(x, rwds, label=label, color=color, marker=marker, 
                    markersize=markersize, lw=linewidth)
            ax.fill_between(x, rwds - std,
                            rwds + std, alpha=0.2,
                                color=color)
            
            
    plt.xlabel('Learning Rate', fontdict={'fontsize': 18})
    # Change tick mark fonts:
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.ylim(1,200_000)
    # plt.yscale('log')

    if value == 'avg_reward':
        plt.ylabel('Average Integrated\nEvaluation Reward (AUC)', fontdict={'fontsize': 18})
        plt.xlabel('Learning rate')

    elif value == 'avg_gap':
        plt.ylabel('Average Gap Between Lower and Upper Bounds')
    plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.525), fancybox=True, 
               shadow=False, ncol=2, fontsize=16)
    # plt.xlim(1e-5,1)
    # use grid lines with sns style:
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlim(0,0.0001)

    #plt.title('Learning Rate Sensitivity')

    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'visualizations/lr_sensitivity{value}_{name}.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':

    for data_name in ['avg_reward', 'avg_gap']:
        fig, ax = plt.subplots(figsize=(6, 5.5))

        csv_files = ['lr_sweep.csv', 'lr_sweep_propalgo_fast.csv',#'lr_sweep2.csv',# 'lr_sweep_propalgo.csv',
                     'mf_lr_sweep_propalgo_fast.csv'] #'lr_sweep_propalgo_fast.csv',]
        csv_files = ['lr_bigmaze.csv']#['big_lr_sweep_propalgo_fast.csv']
        csv_files = ['lr_smallmaze.csv', 'lr_smallmaze_mf.csv']

        for csv_file in csv_files:
            custom_names = None
            custom_symbols = None
            custom_colors = None
            if csv_file == 'lr_sweep_propalgo_fast.csv':#'lr_sweep2.csv':
                custom_names = ['Clipping:\nGiven Model']
            elif csv_file == 'mf_lr_sweep_propalgo_fast.csv':#'lr_sweep_propalgo_fast.csv':
                custom_names = ['Clipping:\nLearned Model']
                custom_symbols = ['*']
                custom_colors = ['g']
            # elif csv_file == 'lr_smallmaze.csv':
                # custom_names = ['Clipping:\nAlgorithm 1']
                # custom_symbols = ['o']
                # # dark red:
                # custom_colors = ['#8B0000']
            elif csv_file == 'lr_smallmaze2.csv':
                custom_names = ['Clipping:\nAlgorithm 2']
                custom_symbols = ['*']
                # bright pink:
                custom_colors = ['#FF007F']
            elif csv_file == 'lr_smallmaze_mf.csv':
                custom_names = ['Clipping:\nLearned Model']
                custom_symbols = ['*']
                custom_colors = ['g']

            plot_all(ax, csv_file, data_name, 
                     custom_names=custom_names,
                     custom_symbols=custom_symbols,
                     custom_colors=custom_colors,
                     name='small2',                     
                     )