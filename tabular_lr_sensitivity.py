import os
import subprocess
import numpy as np

# Sweep over the learning rate, by launching multiple runs of "tabular_lr_singular.py":
methods = {
    'none': {'clip': 'False', 'naive': 'False'},
    'hard': {'clip': 'True', 'naive': 'False'},
    'naive': {'clip': 'True', 'naive': 'True'},
}

# learning_rates = np.logspace(-3, 0, 1000)
# learning_rates = np.linspace(0.0, 0.01, 5)
learning_rates = np.linspace(0.0, 1, 20)


import concurrent.futures
import subprocess

def run_command(command):
    subprocess.run(command, shell=True)


for _ in range(47):

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of commands for each learning rate and method
        commands = [
            f"python tabular_lr_singular.py --lr={lr}" for lr in learning_rates
        # ] + [
        #     f"python tabular_lr_singular.py --clip --naive --lr={lr}" for lr in learning_rates
        # ] + [
            # f"python tabular_lr_singular.py --clip --lr={lr}" for lr in learning_rates
        ]

        # Map the run_command function to the list of commands
        executor.map(run_command, commands)