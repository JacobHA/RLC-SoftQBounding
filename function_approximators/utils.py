import os
import random

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3.common.logger import configure
import time

import torch
import wandb

def env_id_to_envs(env_id, render):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env


def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        # another run may be creating a folder:
        time.sleep(0.5)
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            # another run may be creating a folder:
            time.sleep(0.5)
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"

        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        logger = configure(format_strings=["stdout"])

    return logger

def log_class_vars(self, logger, params, use_wandb=False):
    for key, value in params.items():
        value = self.__dict__[value]
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(key, value)
        if use_wandb:
            wandb.log({key: value})


def sample_wandb_hyperparams(params, int_hparams=None):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] == 'uniform' or v['distribution'] == 'uniform_values':
                sampled[k] = random.uniform(v['min'], v['max'])
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] == 'log_uniform_values':
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                sampled[k] = sample
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if k in int_hparams:
            sampled[k] = int(sampled[k])
    return sampled


def get_bounds(Q, beta, gamma, rewards, mdp_generator, visible_mask=None):
    nS, nA = Q.shape
    delta_rwd = np.empty((nS, nA))
    Qi = Q.flatten()
    Qj = np.log(mdp_generator.T.dot(np.exp(beta * Qi.T)).T) / beta

    delta_rwd = rewards.flatten() + gamma * Qj - Qi
    if visible_mask is not None:
        applicable_deltas = delta_rwd[:, visible_mask]
    else:
        applicable_deltas = delta_rwd

    delta_min, delta_max = np.min(applicable_deltas), np.max(applicable_deltas)
    lb = Qi + delta_rwd + gamma * delta_min / (1-gamma)
    ub = Qi + delta_rwd + gamma * delta_max / (1-gamma)

    # reshape to original shape:
    lb = lb.reshape(nS, nA).A
    ub = ub.reshape(nS, nA).A

    # assert np.allclose(lb <= ub), 'lb > ub'
    return lb, ub

def plot_3d(desc, Q, lb, ub, name_suffix=''):
    """
    Plot the env desc (maze) on a grid. Above this, plot the lb, Q and ub values in 3d.
    """
    # tilt the plot so its easier to see the values vs bounds:
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.view_init(elev=20, azim=30)  # Adjust the elev and azim angles as needed
    # plot the maze:
    nR, nC = desc.shape
    print(desc)
    bar_height = 0.01
    maze_loc = Q.mean() - 0.01*np.abs(Q.mean())
    for i in range(nR):
        for j in range(nC):
            if desc[i, j] == b'H':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='r')
            elif desc[i, j] == b'S':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='g')
            elif desc[i, j] == b'G':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='b')
            elif desc[i, j] == b'F':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='w')
            elif desc[i, j] == b'W':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='k')
            elif desc[i, j] == b'C':
                ax.bar3d(j, i, maze_loc, 1, 1, bar_height, color='y')

    # Calculate the mean Q values for each grid point
    #TODO remove hardcoding:
    nA=4
    q_means = np.mean(Q.reshape(nR, nC, nA), axis=2)
    lb_means = np.mean(lb.reshape(nR, nC, nA), axis=2)
    ub_means = np.mean(ub.reshape(nR, nC, nA), axis=2)

    # Plot the mean Q values as a surface
    x = np.arange(0, nC, 1) + 0.5
    y = np.arange(0, nR, 1) + 0.5
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, q_means, color='k', alpha=0.8, rstride=1, cstride=1)
    ax.plot_surface(x, y, lb_means, color='b', alpha=0.8, rstride=1, cstride=1)
    ax.plot_surface(x, y, ub_means, color='r', alpha=0.8, rstride=1, cstride=1)

    plt.show()
    # plt.pause(0.5)
    # plt.close('all')
    plt.savefig('3d_plot' + name_suffix + '.png')
