from tabular_soln import main

env_str = '7x7zigzag'
oracle = False
gamma = 0.98

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clip', action='store_true')
parser.add_argument('--naive', action='store_true')
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()
clip = args.clip
naive = args.naive
lr = args.lr

rwds = main(env_str=env_str, clip=clip, lr=lr, gamma=gamma, oracle=oracle, naive=naive, save=False)

# Log this run's average reward to a csv file:
import os
import pandas as pd

data = {'clip': [clip], 'naive': [naive], 'lr': [lr], 'avg_reward': [rwds]}

path = 'avg_rewards.csv'
# open the csv in append mode, and add {clip, naive, lr, avg_reward} to the file:
if os.path.exists(path):
    df = pd.read_csv(path)
else:
    df = pd.DataFrame(columns=['clip', 'naive', 'lr', 'avg_reward'])

# add the new data to the df:
df = pd.concat([df, pd.DataFrame(data)], axis=0)
df.to_csv(path, index=False)

