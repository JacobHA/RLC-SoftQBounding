# Open robust_avg_rewards.csv and delete all clip=True naive=False

import pandas as pd
import os

path = 'robust_avg_rewards.csv'
df = pd.read_csv(path)
# Filter out rows where clip=True and naive=False
df = df[~((df['clip'] == True) & (df['naive'] == False))]

# Save the modified DataFrame back to the CSV file if needed
df.to_csv(path, index=False)
