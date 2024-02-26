from collections import deque
import os
import gymnasium
import numpy as np
import pandas as pd

from tabular import ModifiedFrozenLake, get_dynamics_and_rewards, get_mdp_generator, visible_states_mask
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt

class SoftQLearning():
    def __init__(self, env, beta, gamma, learning_rate_schedule, 
                 init_Q=None, prior_policy=None, plot=False, 
                 save_data=False, clip=False, lb=None, ub=None,
                 prefix='', keep_bounds_fixed=False):
        self.env = env
        self.nS = env.observation_space.n
        self.nA = env.action_space.n

        self.beta = beta
        self.gamma = gamma
        self.learning_rate_schedule = learning_rate_schedule
        self.clip = clip
        self.save_data = save_data
        if save_data:
            prefix = prefix+'clip-'*bool(clip)+str(gamma)
            # count how many files are in data folder:
            if not os.path.exists(f'{prefix}data'):
                os.makedirs(f'{prefix}data')
            num = len(os.listdir(f'{prefix}data')) + 1
            # append a number to the filename:
            path = f'{prefix}data/softq_{num}.npy'
            # if path exists, increment the number:
            while os.path.exists(path):
                num += 1
                path = f'{prefix}data/softq_{num}.npy'
            print(f'Saving data to {path}')
            self.path = path

        if init_Q is not None:
            assert init_Q.shape == (self.nS, self.nA), f'init_Q.shape: {init_Q.shape}, expected shape: {(self.nS, self.nA)}'
            self.Q = init_Q
        else:
            # random initialization:
            self.Q = np.random.rand(self.nS, self.nA) #/ (1 - self.gamma)
            # self.Q = np.ones((self.nS, self.nA)) * 0.5 / (1 - self.gamma)

        if prior_policy is None:
            self.prior_policy = 1/self.nA * np.ones((self.nS, self.nA))
        else:
            self.prior_policy = prior_policy


        self.plot = plot
        self.keep_bounds_fixed = keep_bounds_fixed
        # get rewards and dynamics:

        self.visible_mask, _ = visible_states_mask(self.env.desc)
        self.dynamics, self.rewards = get_dynamics_and_rewards(env)
        self.mdp_generator = get_mdp_generator(self.env, self.dynamics, self.prior_policy)

        self.rewards = self.rewards.reshape(self.nS, self.nA)
        self.lb = np.ones((self.nS, self.nA)) * -np.inf if lb is None else lb
        self.ub = np.ones((self.nS, self.nA)) * np.inf if ub is None else ub
        if not keep_bounds_fixed:
            self.lb, self.ub = self.get_bounds()
        self.total_clips = 0
        self.Q_over_time = []
        self.q_stds = []
        self.lb_over_time = []
        self.ub_over_time = []
        self.lb_stds = []
        self.ub_stds = []
        self.reward_over_time = []
        self.loss_over_time = []


    def V_from_Q(self, Q):
        # first get a baseline for numerical stability:
        b = (np.max(Q) + np.min(Q))/2
        Q = Q - b
        # compute soft max wrt actions and prior policy
        V = 1/self.beta * np.log(np.sum(np.exp(self.beta * Q) * self.prior_policy, axis=1))
        # V = E_pi( Q + 1/beta log pi(a'|s'))
        policy = self.pi_from_Q(Q, V)
        # V = np.dot(policy, Q + 1/self.beta * np.log(policy))
        return V + b
    
    def pi_from_Q(self, Q, V=None):
        if V is None:
            V = self.V_from_Q(Q)

        pi = self.prior_policy * np.exp(self.beta * (Q - V.reshape(-1, 1)))
        pi = pi / np.sum(pi, axis=1, keepdims=True)
        return pi
    
    def draw_action(self, pi, state, greedy=False):
        if greedy:
            return np.argmax(pi[state])
        else:
            return np.random.choice(self.nA, p=pi[state[0]])
        
    def learn(self, state, action, reward, next_state, done, lr):
        # Compute the TD error:
        next_V = self.V_from_Q(self.Q)[next_state]
        target = reward + (1 - done) * self.gamma * next_V
        delta = target - self.Q[state, action]
        # Update the Q value:
        self.Q[state, action] += lr * delta
        if self.clip:
            # self.Q[state, action] = np.clip(self.Q[state, action], self.lb[state, action], self.ub[state, action])
            self.Q = np.minimum(np.maximum(self.Q, self.lb), self.ub)

            # count how many values were clipped:
            self.total_clips += np.sum(self.Q[state, action] == self.lb[state, action]) + np.sum(self.Q[state, action] == self.ub[state, action])

        return delta
    
    def train(self, max_steps, render=False, greedy_eval=False, eval_freq=100):
        self.times = np.arange(max_steps, step=eval_freq)
        if self.plot:
            plt.ion()
            plt.show()
            # add a second y axis for error and second panel for q values:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
            self.ax11 = self.ax1.twinx()
            self.ax1.set_xlabel('Environment Steps')
            self.ax1.set_ylabel('Evaluation reward', color='b')
            self.ax11.set_ylabel('TD error', color='r')
            self.ax11.set_yscale('log')
            self.ax2.set_xlabel('Environment Steps')
            self.ax2.set_ylabel('Q values')

        state, _ = self.env.reset()
        steps = 0
        total_reward = 0
        done = False
        while steps < max_steps:
            pi = self.pi_from_Q(self.Q)
            action = self.draw_action(pi, state, greedy=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            lr = self.learning_rate_schedule(steps)
            delta = self.learn(state, action, reward, next_state, terminated, lr)
            state = next_state
            steps += 1
            if render:
                self.env.render()
            if done:
                state, _ = self.env.reset()
                done = False
                # update bounds:
                if not self.keep_bounds_fixed:
                    self.lb, self.ub = self.get_bounds()

            if steps % eval_freq == 0:
                if not self.keep_bounds_fixed:
                    self.lb, self.ub = self.get_bounds()

                eval_rwd = self.evaluate(1, render=False, greedy=greedy_eval)
                total_reward += eval_rwd
                print(f'steps={steps}, eval_rwd={eval_rwd:.2f}, lb={self.lb.mean():.2f}, ub={self.ub.mean():.2f}, lr={lr:.6f}')
                if self.plot:
                    self.live_plot(eval_rwd, steps, error=np.abs(delta))
                if self.save_data:
                    self.Q_over_time.append(self.Q.mean())
                    self.lb_over_time.append(self.lb.mean())
                    self.ub_over_time.append(self.ub.mean())
                    self.reward_over_time.append(eval_rwd)
                    self.loss_over_time.append(np.abs(delta)[0])
                    # Save mins and maxs of lbs and ubs and Q:
                    self.lb_stds.append(self.lb.max() - self.lb.min())
                    self.ub_stds.append(self.ub.max() - self.ub.min())
                    self.q_stds.append(self.Q.std())
            
                    # Save the data:
                    self.save()

            # if steps % 10_000 == 0:
                # plot_3d(self.env.desc, self.Q, self.lb, self.ub)


        return total_reward
    
    
    def evaluate(self, num_episodes, render=False, greedy=False):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                # print(self.total_clips)
                pi = self.pi_from_Q(self.Q)
                action = self.draw_action(pi, state, greedy)
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if render:
                    self.env.render()
                done = terminated or truncated
        return total_reward / num_episodes
    
    def live_plot(self, reward, step, error=None):
        # update the figure by adding the next point:
        self.ax1.plot(step, reward, 'bo-')
        if error is not None:
            self.ax11.plot(step, error, 'ro')
            
        # Plot the mean q value and lb and ub:
        q = self.Q.flatten()[self.visible_mask]
        l = self.lb.flatten()[self.visible_mask]
        u = self.ub.flatten()[self.visible_mask]

        mean_Q = np.mean(q)
        max_Q, min_Q = np.max(q), np.min(q)
        
        self.ax2.plot(step, mean_Q, 'go')
        # Error bar for max and min of Q:
        self.ax2.errorbar(step, mean_Q, yerr=[[mean_Q - min_Q], [max_Q - mean_Q]], fmt='go')

        lb, lb_min, lb_max = np.mean(l), np.min(l), np.max(l)
        ub, ub_min, ub_max = np.mean(u), np.min(u), np.max(u)

        self.ax2.plot(step, lb, 'bo')
        self.ax2.errorbar(step, lb, yerr=[[lb - lb_min], [lb_max - lb]], fmt='bo')

        self.ax2.plot(step, ub, 'ro')
        self.ax2.errorbar(step, ub, yerr=[[np.abs(ub - ub_min)], [np.abs(ub_max - ub)]], fmt='ro')

        # Update the plot and pause:
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def get_bounds(self):
        delta_rwd = np.empty((self.nS, self.nA))
        Qi = self.Q.flatten()
        Qj = np.log(self.mdp_generator.T.dot(np.exp(self.beta * Qi.T)).T) / self.beta
 
        delta_rwd = self.rewards.flatten() + self.gamma * Qj - Qi
        applicable_deltas = delta_rwd[:, self.visible_mask]

        delta_min, delta_max = np.min(applicable_deltas), np.max(applicable_deltas)
        lb = Qi + delta_rwd + self.gamma * delta_min /(1-self.gamma)
        ub = Qi + delta_rwd + self.gamma * delta_max /(1-self.gamma)

        # reshape to original shape:
        lb = lb.reshape(self.nS, self.nA).A
        ub = ub.reshape(self.nS, self.nA).A

        # take the tighter bound from previous step:
        lb = np.maximum(lb, self.lb)
        ub = np.minimum(ub, self.ub)

        # assert np.allclose(lb <= ub), 'lb > ub'
        return lb, ub
                
    def save(self):
        # Save the data with np:
        # calculate the timesteps data:
        data = np.array([self.Q_over_time, self.q_stds, self.lb_over_time, self.lb_stds, self.ub_over_time, self.ub_stds,
                         self.reward_over_time, self.loss_over_time, self.times[:len(self.Q_over_time)]])
        np.save(self.path, data)
        
        

def plot_3d(desc, Q, lb, ub):
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

    # plot the Q values:
    # for i in range(nR):
    #     for j in range(nC):
    #         # for a in range(4):
    #         q = Q[i*nC + j, :].mean()
    #         l = lb[i*nC + j, :].mean()
    #         u = ub[i*nC + j, :].mean()
    #         ax.bar3d(j, i, q, 1, 1, bar_height, color='k', alpha=0.2)
    #         # ax.bar3d(j, i, l, 1, 1, bar_height, color='b', alpha=0.2)
    #         ax.bar3d(j, i, u, 1, 1, bar_height, color='r', alpha=0.2)

    # Calculate the mean Q values for each grid point
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
    plt.pause(0.5)
    plt.close('all')


def main(env_str, clip, gamma, oracle, naive, save=True, lr=None):
    # 11x11dzigzag
    env = ModifiedFrozenLake(map_name=env_str,cyclic_mode=False,slippery=0)
    env = TimeLimit(env, max_episode_steps=1000)
    # env = gymnasium.make('FrozenLake-v1', is_slippery=False)
    beta = 5
    gamma = 0.98
    lb, ub = None, None
    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nA = env.observation_space.n, env.action_space.n
    if oracle:
        # solve the MDP exactly:
        from tabular import softq_solver
        Q, _, _ = softq_solver(env, beta=beta, gamma=gamma, tolerance=1e-14)
        # calculate bounds
        from utils import get_bounds
        generator = get_mdp_generator(env, dynamics, np.ones((nS, nA)) / nA)
        lb, ub = get_bounds(Q, beta, gamma, rewards, generator)

    if naive:
        # Get naive bounds rmin and rmax over 1-gamma:
        lb = np.min(rewards) / (1 - gamma) * np.ones((nS, nA))
        ub = np.max(rewards) / (1 - gamma) * np.ones((nS, nA))
    
    def learning_rate_schedule(t):
        if lr is None:
            if clip:
                if naive:
                    return 0.65
                else:
                    return 0.15
            else:
                return 0.7
        else:
            return lr

    sarsa = SoftQLearning(env, beta, gamma, learning_rate_schedule,
                           plot=0, save_data=save, clip=clip, lb=lb, ub=ub,
                           prefix='oracle'*oracle+'naive'*naive,
                           keep_bounds_fixed=naive)
    max_steps = 100_000

    total_reward = sarsa.train(max_steps, render=False, greedy_eval=True, eval_freq=100)
    return total_reward


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='7x7zigzag')
    parser.add_argument('--clip', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--oracle', type=bool, default=False)
    parser.add_argument('--naive', type=bool, default=False)
    parser.add_argument('-n', type=int, default=1)
    args = parser.parse_args()

    for _ in range(args.n):
        main(env_str=args.env, clip=args.clip, gamma=args.gamma, oracle=args.oracle, naive=args.naive)