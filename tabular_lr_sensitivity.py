import os
import subprocess
import numpy as np

runs_per_hparam = 5

# Sweep over the learning rate, by launching multiple runs of "tabular_lr_singular.py":
methods = {
    'none': {'clip': 'False', 'naive': 'False'},
    'hard': {'clip': 'True', 'naive': 'False'},
    'naive': {'clip': 'True', 'naive': 'True'},
}

# learning_rates = np.logspace(-3, 0, 1000)
# learning_rates = np.linspace(0.0, 0.01, 5)
learning_rates = np.linspace(0.0, 0.0001, 10)



for lr in learning_rates:
    # Launch a run for each method and 5 runs for each learning rate:

    command1 = f"python tabular_lr_singular.py --lr={lr}"
    command2 = f"python tabular_lr_singular.py --clip --naive --lr={lr}"
    command3 = f"python tabular_lr_singular.py --clip --lr={lr}"

    subprocess.run(command3, shell=True)
    subprocess.run(command2, shell=True)
    subprocess.run(command1, shell=True)