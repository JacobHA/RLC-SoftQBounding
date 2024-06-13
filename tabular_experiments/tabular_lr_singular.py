from proposed_soln import main_sweep
import numpy as np
import concurrent.futures
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--clip', action='store_true')
parser.add_argument('--naive', action='store_true')
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()
clip = args.clip
naive = args.naive
lr = args.lr
path = 'lr_smallmaze.csv'


# open the csv in append mode, and add {clip, naive, lr, avg_reward} to the file:
if os.path.exists(path):
    df = pd.read_csv(path)
else:
    df = pd.DataFrame(columns=['clip', 'naive', 'lr', 'avg_reward', 'avg_gap'])

for num in range(3,10):
    # Grab a map from the random mazes folder
    desc = np.load(f'mazes/random_mazes/random_map_{num}.npy')
    desc = list(desc)

    def process_map(_):
        return main_sweep(desc, lr=lr, give_model=True, clip=clip)#, naive=naive)

    # Number of random maps
    n_random_maps = 1

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Map the process_map function to the range of n_random_maps
        results = list(executor.map(process_map, range(n_random_maps)))

    # results = process_map(0)

    # Separate rewards and gaps
    rewards = [result[0] for result in results]
    gaps = [result[1] for result in results]

    # Calculate the average reward and gaps
    average_reward = sum(rewards) / n_random_maps
    average_gaps = sum(gaps) / n_random_maps

    # Log this run's average reward to a csv file:


    data = {'clip': [clip], 'naive': [naive], 'lr': [lr], 'avg_reward': [average_reward], 'avg_gap': [average_gaps]}

    # add the new data to the df:
    df = pd.concat([df, pd.DataFrame(data)], axis=0)
    df.to_csv(path, index=False)

