from collections import deque
import os
import gymnasium
import numpy as np
import pandas as pd
import sys

from utils import get_optimal_reward, get_random_policy_reward
sys.path.append('visualizations')

from tabular import ModifiedFrozenLake, get_dynamics_and_rewards, get_mdp_generator, visible_states_mask
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt

class NewSoftQLearning():
    def __init__(self, env, beta, gamma, learning_rate_schedule, 
                 init_Q=None, prior_policy=None, plot=False, 
                 save_data=False, clip=False, lb=None, ub=None,
                 prefix='', keep_bounds_fixed=False, give_model=False,
                 verbose=False):
        self.env = env
        self.rewards_table = -np.ones((env.observation_space.n, env.action_space.n))
        self.dynamics_table = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.verbose = verbose

        self.beta = beta
        self.gamma = gamma
        self.learning_rate_schedule = learning_rate_schedule
        self.clip = clip
        self.give_model = give_model
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
        # if not keep_bounds_fixed:
        #     self.lb, self.ub = self.get_bounds()
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
        # policy = self.pi_from_Q(Q, V)
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
        # self.Q = np.minimum(np.maximum(self.Q, self.lb), self.ub)
        next_V = self.V_from_Q(self.Q)[next_state]

        target = reward + (1 - done) * self.gamma * next_V
        bellman_diff = target - self.Q[state, action]
        self.Q[state,action] += bellman_diff * lr

        # self.Q[state, action] = np.clip(self.Q[state, action], self.lb[state, action], self.ub[state, action])
        if self.clip:
            self.Q = np.minimum(np.maximum(self.Q, self.lb), self.ub)

        # # check if Q has changes:
        # if np.allclose(Qnew, self.Q):
        #     # use bellman
        #     self.Q[state,action] += bellman_diff * lr

        # else:
        #     # use the clipped Q:
        #     self.Q = Qnew

        # # count how many values were clipped:
        self.total_clips += np.sum(self.Q[state, action] == self.lb[state, action]) + \
            np.sum(self.Q[state, action] == self.ub[state, action])

        return bellman_diff
    
    def train(self, max_steps, render=False, greedy_eval=False, 
              eval_freq=100, bound_update_freq=None):
        if bound_update_freq is not None:
            self.bound_update_freq = bound_update_freq
        else:
            self.bound_update_freq = eval_freq
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
            # Update the reward and dynamics tables:
            if not self.give_model:
                # update the rewards and dynamics tables:
                self.rewards_table[state, action] = reward
                self.dynamics_table[state, action, next_state] += 1

                dyn = self.dynamics_table.reshape(self.nS*self.nA, self.nS).T
                # normalize the nonzero columns:
                for i in range(dyn.shape[1]):
                    if np.sum(dyn[:,i]) > 0:
                        dyn[:,i] /= np.sum(dyn[:,i])

            done = terminated or truncated
            lr = self.learning_rate_schedule(steps)
            # first learn a sufficiently accurate model:
            if self.give_model:
                if steps % self.bound_update_freq == 0:
                    self.lb, self.ub = self.get_bounds()
            else:    
                if steps > 5000:
                    if steps % self.bound_update_freq == 0:
                        self.lb, self.ub = self.get_bounds()

            delta = self.learn(state, action, reward, next_state, terminated, lr)
            state = next_state
            steps += 1
            if render:
                self.env.render()
            if done:
                state, _ = self.env.reset()
                done = False

            if steps % eval_freq == 0:
                eval_rwd = self.evaluate(1, render=False, greedy=greedy_eval)
                total_reward += eval_rwd
                if self.verbose:
                    print(f'steps={steps}, eval_rwd={eval_rwd:.2f}, lb={self.lb.mean():.2f}, ub={self.ub.mean():.2f}, lr={lr:.3f}, loss={np.abs(delta)[0]:.3}')
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
        self.ax2.errorbar(step, lb, yerr=[[np.abs(lb - lb_min)], [np.abs(lb_max - lb)]], fmt='bo')

        self.ax2.plot(step, ub, 'ro')
        self.ax2.errorbar(step, ub, yerr=[[np.abs(ub - ub_min)], [np.abs(ub_max - ub)]], fmt='ro')

        # Update the plot and pause:
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.01)

    def get_bounds(self):
        from scipy.sparse import csr_matrix

        delta_rwd = np.empty((self.nS, self.nA))
        Qi = self.Q.flatten()
        if self.give_model:
            dyn = self.mdp_generator
        else:
            # calculate mdp generator from the dynamics table:
            p = self.dynamics_table.reshape(self.nS * self.nA, self.nS)
            dyn = get_mdp_generator(self.env, 
                            csr_matrix(p.T), 
                            self.prior_policy)

        # Qj = np.log(self.mdp_generator.T.dot(np.exp(self.beta * Qi.T)).T) / self.beta
        baseline = (np.mean(Qi) + np.min(Qi)) / 2
        Qi -= baseline
        Qj = np.log(dyn.T.dot(np.exp(self.beta * Qi.T)).T) / self.beta + baseline
        Qi += baseline
        # ignore the Qj's that are inf:
        finite = np.where(np.isfinite(Qj) & np.isfinite(Qi))
        # mask with the visible mask:
        vis_and_finite = np.intersect1d(finite, self.visible_mask)


        if self.give_model:
            delta_rwd = self.rewards.A.flatten() + self.gamma * Qj - Qi
        else:
            delta_rwd = self.rewards_table.flatten() + self.gamma * Qj - Qi

        applicable_deltas = delta_rwd[vis_and_finite]
        # ignore the inf deltas:
        delta_rwd[~np.isfinite(delta_rwd)] = np.ones(len([~np.isfinite(delta_rwd)])) * \
            np.mean(delta_rwd[vis_and_finite])

        delta_min, delta_max = np.min(applicable_deltas), np.max(applicable_deltas)
        lb = Qi + delta_rwd + self.gamma * delta_min / (1 - self.gamma)
        ub = Qi + delta_rwd + self.gamma * delta_max / (1 - self.gamma)

        # reshape to original shape:
        lb = lb.reshape(self.nS, self.nA)
        ub = ub.reshape(self.nS, self.nA)

        if self.give_model:
            # We can trust the bounds from the model, 
            # so take the tighter bound from previous step:
            lb = np.maximum(lb, self.lb)
            ub = np.minimum(ub, self.ub)
        else:    
            lb = np.maximum(lb, self.rewards.min() / (1 - self.gamma))
            ub = np.minimum(ub, self.rewards.max() / (1 - self.gamma))

        assert np.all(lb <= ub), 'lb > ub'
        return lb, ub
                
    def save(self):
        # Save the data with np:
        # calculate the timesteps data:
        data = np.array([self.Q_over_time, self.q_stds, self.lb_over_time, self.lb_stds, self.ub_over_time, self.ub_stds,
                         self.reward_over_time, self.loss_over_time, self.times[:len(self.Q_over_time)]])
        np.save(self.path, data)
        
nS_to_settings = {
    49: {'gamma': 0.98, 'max_steps': 200_000, 'eval_freq': 100, 'bound_update_freq': 10},
    144: {'gamma': 0.98, 'max_steps': 600_000, 'eval_freq': 300, 'bound_update_freq': 30},
}

def main_sweep(map_desc, lr, give_model, clip, plot=False):

    print("Clip:", clip)

    env = ModifiedFrozenLake(desc=map_desc, cyclic_mode=False, slippery=0.5)
    env = TimeLimit(env, max_episode_steps=1000)
    nS, nA = env.observation_space.n, env.action_space.n

    beta = 5
    gamma = nS_to_settings[nS]['gamma']
    _, rewards = get_dynamics_and_rewards(env)
    
    optimal_reward, Q = get_optimal_reward(env, beta, gamma)
    print(f'Optimal reward: {optimal_reward}')
    worst_reward = get_random_policy_reward(env, nA)

    # Get naive bounds rmin and rmax over 1-gamma:
    lb = np.min(rewards) / (1 - gamma) * np.ones((nS, nA))
    ub = np.max(rewards) / (1 - gamma) * np.ones((nS, nA))

    # Loop over the same random maze (stochastic dynamics):
    normalized_reward = 0
    normalized_gap = 0
    num_runs = 1
    for run in range(num_runs):

        sarsa = NewSoftQLearning(env, beta, gamma, lambda t: lr,
                                save_data=False, clip=clip, lb=lb, ub=ub,
                                keep_bounds_fixed=False,
                                give_model=give_model,
                                plot=plot)
        
        max_steps = nS_to_settings[nS]['max_steps']
        eval_freq = nS_to_settings[nS]['eval_freq']
        bound_update_freq = nS_to_settings[nS]['bound_update_freq']
        total_reward = sarsa.train(max_steps, render=False, greedy_eval=True, 
                                   eval_freq=eval_freq, bound_update_freq=bound_update_freq)
        
        # normalize the reward by the optimal reward and worst-case:
        normalized_reward += (total_reward / (max_steps // eval_freq) - worst_reward) / (optimal_reward - worst_reward)
        # Get the final (average) gap between lower and upper bound:
        gap = np.mean(sarsa.ub - sarsa.lb)
        normalized_gap += gap / np.mean(Q)

        print(f'Run {run+1}/{num_runs} completed. Reward: {total_reward / (max_steps // eval_freq)}')

    return normalized_reward / num_runs, normalized_gap / num_runs



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='random')
    parser.add_argument('--clip', type=bool, default=False)
    parser.add_argument('--oracle', type=bool, default=False)
    parser.add_argument('--naive', type=bool, default=False)
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    desc = list(np.load(f'mazes/random_mazes/random_map_0.npy'))

    for _ in range(args.n):
        main_sweep(desc, args.lr, clip=args.clip, give_model=True, plot=True)