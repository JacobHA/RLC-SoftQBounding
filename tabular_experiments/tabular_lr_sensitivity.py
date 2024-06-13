import subprocess
import numpy as np
import concurrent.futures

# Sweep over the learning rate, by launching multiple runs of "tabular_lr_singular.py":
methods = {
    'none': {'clip': 'False', 'naive': 'False'},
    'hard': {'clip': 'True', 'naive': 'False'},
    'naive': {'clip': 'True', 'naive': 'True'},
}

learning_rates = np.logspace(-5, 0, 20)

def run_command(command):
    subprocess.run(command, shell=True)

max_workers = 24
workers = min(max_workers, len(learning_rates))

# Using ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    # Create a list of commands for each learning rate and method
    commands = [
        f"python tabular_lr_singular.py --lr={lr}" for lr in learning_rates
    ] + [
        f"python tabular_lr_singular.py --clip --naive --lr={lr}" for lr in learning_rates
    ] + [
        f"python tabular_lr_singular.py --clip --lr={lr}" for lr in learning_rates
    ]

    # Map the run_command function to the list of commands
    executor.map(run_command, commands)