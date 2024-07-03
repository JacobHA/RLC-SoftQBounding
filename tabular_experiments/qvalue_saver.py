# Solve FrozenLake-v1 and save the Q values:

import gymnasium as gym
import numpy as np
from tabular import get_dynamics_and_rewards, softq_solver, get_mdp_generator
from utils import get_bounds

def solve_frozenlake():
    env = gym.make('FrozenLake-v1')
    beta = 1.0
    gamma = 0.99
    Q, _, _ = softq_solver(env, beta=beta, gamma=gamma, tolerance=1e-14)
    print(Q)
    # get bounds:
    dynamics, rewards = get_dynamics_and_rewards(env)
    # get mdp generator:
    pi0 = np.ones((env.nS, env.nA)) / env.nA
    mdp_generator = get_mdp_generator(env, dynamics, pi0)

    lb, ub = get_bounds(Q, beta, gamma, rewards, mdp_generator)
    print(lb)
    print(ub)

    np.savez('frozenlake.npz', Q=Q, lb=lb, ub=ub)

if __name__ == '__main__':
    solve_frozenlake()