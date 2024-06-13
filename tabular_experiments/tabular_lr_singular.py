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

# check that the csv file exists:
if not os.path.exists(path):
    with open(path, 'w') as f:
        f.write('clip,naive,lr,avg_reward,avg_gap\n')

else:
    # ensure the first row is the column names:
    df = pd.read_csv(path)
    if 'clip' not in df.columns:
        with open(path, 'w') as f:
            f.write('clip,naive,lr,avg_reward,avg_gap\n')

for num in range(30):
    print("Running random map", num)
    # Grab a map from the random mazes folder
    desc = np.load(f'mazes/random_mazes/random_map_{num}.npy')
    desc = list(desc)

    def process_map(_):
        return main_sweep(desc, lr=lr, give_model=True, clip=clip, naive=naive)

    # Number of random maps
    n_random_maps = 1

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Map the process_map function to the range of n_random_maps
        results = list(executor.map(process_map, range(n_random_maps)))

    # Separate rewards and gaps
    rewards = [result[0] for result in results]
    gaps = [result[1] for result in results]

    # Calculate the average reward and gaps
    average_reward = sum(rewards) / n_random_maps
    average_gaps = sum(gaps) / n_random_maps

    # Log this run's average reward to a csv file:


    data = {'clip': [clip], 'naive': [naive], 'lr': [lr], 'avg_reward': [average_reward], 'avg_gap': [average_gaps]}
    # add the new data to the most up-to-date df:
    df = pd.read_csv(path)
    df = pd.concat([df, pd.DataFrame(data)], axis=0)
    df.to_csv(path, index=False)


