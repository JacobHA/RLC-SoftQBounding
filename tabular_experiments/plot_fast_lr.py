import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define a mapping for methods
methods = {
    'hard': {'clip': True, 'naive': False},
    'naive': {'clip': True, 'naive': True},
    'none': {'clip': False, 'naive': False},
}

def preprocess(csv_file):
    # Load the data from avg_rewards.csv:
    df = pd.read_csv(csv_file)

    # Define a function to calculate std error of the mean
    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    # Get the number of unique learning rates:
    lrs = df['lr'].unique()

    # apply the func auc_to_step to the [value]:
    df['avg_reward'] = df['avg_reward'].apply(auc_to_step)


    # Aggregate by group, calculating mean and SEM
    result = df.groupby(['lr', 'clip', 'naive']).agg(['mean', sem, 'count'])

    # Rename columns for clarity
    # result.columns = ['Mean', 'SEM']

    # df = df.groupby(['lr', 'clip', 'naive']).agg({'avg_reward': ['mean', 'std']})
    df = result.reset_index()
    return df

def auc_to_step(auc, max_step=200):
    # Convert AUC to step function:
    return 500*(1 - auc)

def plot_all(ax, csv_file, value, custom_names=None, custom_symbols=None):

    df = preprocess(csv_file)

    # Plot each of the methods in a different color:
    if custom_names is not None:
        # assert len(custom_names) == len(), "Custom names must have the same length as the methods dict."
        labels = custom_names
    else:
        labels = ['Clipping:\nProposed\nBounds', "Clipping:\nBaseline\nBounds", 'No Clipping']

    if custom_symbols is not None:
        markers = custom_symbols

    colors = ['#FF5733', '#007acc',  '#333333']
    markers = ['o', '^', 's']

    # if not csv_file in ['lr_sweep2.csv', 'lr_sweep_propalgo.csv']:
    #     # Remove the hard clip method from plotting:
    #     labels = labels[1:]
    #     colors = colors[1:]
    #     markers = markers[1:]

    for method, label, color, marker in zip(methods.keys(), labels, colors, markers):
        clip = methods[method]["clip"]
        naive = methods[method]["naive"]

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
            linewidth = 4
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
    if value == 'avg_reward':
        plt.ylabel('Average Integrated\nEvaluation Reward (AUC)', fontdict={'fontsize': 18})
        plt.xlabel('Learning rate')

    elif value == 'avg_gap':
        plt.ylabel('Average Gap Between Lower and Upper Bounds')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.325), fancybox=True, 
               shadow=False, ncol=3, fontsize=16)
    # plt.xlim(1e-5,1)
    # use grid lines with sns style:
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.xlim(0,0.0001)

    #plt.title('Learning Rate Sensitivity')
    # plt.ylim(-0.1,1.05)

    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'visualizations/big_fast_lr_sensitivity{value}.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':

    for data_name in ['avg_reward', 'avg_gap']:
        fig, ax = plt.subplots(figsize=(8, 6))

        csv_files = ['lr_sweep_propalgo_fast.csv']

        for csv_file in csv_files:
            custom_names = None
            if csv_file == 'big_lr_sweep_propalgo_fast.csv':
                custom_names = ['Clipping:\nProposed\nAlgorithm']
                custom_symbols = ['*']
            plot_all(ax, csv_file, data_name, custom_names=custom_names)
            if data_name == 'avg_reward':
                plt.ylim(-0.1,1.05)

