"""Customized Frozen lake enviroment"""
import sys
from contextlib import closing
from gymnasium.envs.toy_text import FrozenLakeEnv
from gymnasium import utils
from joblib import Parallel, delayed
import numpy as np
import os
import pickle 
from six import StringIO

import itertools
import gymnasium
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

def chi(u, n_states, n_actions, prior_policy=None):
    if prior_policy is None:
        prior_policy = np.ones((n_states, n_actions)) / n_actions
    u = u.reshape(n_states, n_actions)
    return (prior_policy * u).sum(axis=1)


def get_dynamics_and_rewards(env):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = dynamics.sum(axis=0)
    assert (colsums.round(12) == 1.).all(), f"{colsums.min()}, {colsums.max()}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)),
                         shape=shape).sum(axis=0)

    return dynamics, rewards


def find_exploration_policy(dynamics, rewards, n_states, n_actions, beta=1, alpha=0.01, prior_policy=None, debug=False, max_it=20):

    rewards[:] = 0
    prior_policy = np.matrix(np.ones((n_states, n_actions))) / \
        n_actions if prior_policy is None else prior_policy
    if debug:
        entropy_list = []

    for i in range(1, 1 + max_it):
        u, v, optimal_policy, _, estimated_distribution, _ = solve_biased_unconstrained(
            beta, dynamics, rewards, prior_policy, bias_max_it=20)

        sa_dist = np.multiply(u, v.T)
        mask = sa_dist > 0
        r = rewards.copy()
        r[:] = 0.
        r[mask] = - np.log(sa_dist[mask].tolist()[0]) / beta
        r = r - r.max()
        rewards = (1 - alpha) * rewards + alpha * r

        if debug:
            x = sa_dist[sa_dist > 0]
            entropy = - np.multiply(x, np.log(x)).sum()
            entropy_list.append(entropy)

            # print(f"{i=}\t{alpha=:.3f}\t{entropy=: 10.4f}\t", end='')

    return optimal_policy


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    tolerance *= beta

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()
    M_scale = 1.

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1)))
    u_scale = np.sum(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * nSnA ** 2
    v_scale = np.sum(v)

    lol = float('inf')
    hil = 0.

    for i in range(1, eig_max_it+1):

        uk = (Mt).dot(u)
        lu = np.sum(uk) / u_scale
        mask = np.logical_and(uk > 0., uk < np.inf)
        rescale = 1. / np.sqrt(uk[mask].max()*uk[mask].min())
        uk = uk / lu * rescale
        u_scale *= rescale

        vk = M.dot(v)
        lv = np.sum(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))
                       ).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))
                       ).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100 == 0:
            rescale = 1 / np.sqrt(lu)
            Mt = Mt * rescale
            M_scale *= rescale

        if u_err <= tolerance and v_err <= tolerance:
            # if u_err <= tolerance:
            l = lu / M_scale
            # print(f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu / M_scale
        # : {i: 8d}, {u.min() = : .4e}, {u.max() = : .4e}. {M_scale = : .4e}, {lu = : .4e}, {l = : .4e}, {u_err = : .4e}, {v_err = : .4e}")
        print(f"Did not converge")

    l = lu / M_scale

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(
        u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def visible_states_mask(desc, nact=4):
    env = ModifiedFrozenLake(desc, n_action=nact)
    dynamics, _ = get_dynamics_and_rewards(env)
    dynamics = dynamics.A.T
    invis = []
    for num, row in enumerate(dynamics.T):
        if row.sum() < 4:
            invis.append(num)
    # These are states that cannot be reached by any action
    # copy them to all state-action pairs:
    invis = np.array(invis)
    invis = sorted(np.concatenate([invis*nact + i for i in range(nact)]))
    # Now get the visible states, those that can be transitioned into
    vis = np.array(
        [i for i in range(dynamics.shape[0]) if i not in invis])

    return vis, invis


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA +
            np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)),
                           (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows, cols)), shape=(
        nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix

def training_episode(env, training_policy):
    sarsa_experience = []

    state = env.reset()
    action = np.random.choice(env.nA, p=training_policy[state])
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = np.random.choice(env.nA, p=training_policy[next_state])
        sarsa_experience.append(
            ((state, action, reward, next_state, next_action), done))
        state, action = next_state, next_action

    return sarsa_experience


def gather_experience(env, training_policy, batch_size, n_jobs=1):
    if n_jobs > 1:
        split_experience = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(training_episode)(env, training_policy) for _ in range(batch_size))
    elif n_jobs == 1:
        split_experience = [training_episode(
            env, training_policy) for _ in range(batch_size)]

    return list(itertools.chain.from_iterable(split_experience))

# From old gym code for DiscreteEnv:
def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()

class DiscreteEnv(gymnasium.Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, False, {"prob": p})



def get_dynamics_and_rewards(env):
    if not hasattr(env, 'nA'):
        env.nA = env.action_space.n
    if not hasattr(env, 'nS'):
        env.nS = env.observation_space.n
    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = dynamics.sum(axis=0)
    assert (colsums.round(12) == 1.).all(
    ), f"{colsums.min()=}, {colsums.max()=}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)),
                         shape=shape).sum(axis=0)

    return dynamics, rewards


def softq_solver(env, prior_policy=None, steps=100_000, beta=1, gamma=1, tolerance=1e-2, savename=None, verbose=False, rewards=None, resolve=False, Q0=None):

    # First we check if the solution is already saved somewhere, if so, we load it
    if savename is not None and not resolve:
        try:
            with open(savename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

    if rewards is None:
        dynamics, rewards = get_dynamics_and_rewards(env)
    else:
        dynamics, _ = get_dynamics_and_rewards(env)

    errors_list = []
    qs = []

    rewards = dynamics.multiply(rewards).sum(axis=0)
    prior_policy = np.ones((env.nS, env.nA)) / \
        env.nA if prior_policy is None else prior_policy
    mdp_generator = get_mdp_generator(env, dynamics, prior_policy)

    if Q0 is None:
        Qi = np.zeros((1, env.nS * env.nA))
        Qi = np.random.rand(1, env.nS * env.nA) * rewards.max() / \
            (1-gamma)  # the approximate scale of Q
    else:
        Qi = Q0
        Qi = Qi.reshape((1, env.nS * env.nA))

    for i in range(1, steps+1):
        baseline = Qi.mean()
        Qj = np.log(mdp_generator.T.dot(np.exp(beta * (Qi.T - baseline))).T) / beta
        Qj += baseline
        Qi_k = rewards.A + gamma * Qj
        if verbose:
            
            qs.append(Qi_k.mean())

        err = np.abs(Qi_k - Qi).max() # L_infty norm
        # err = np.abs(Qi_k - Qi).sum() # L_1 norm
        Qi = Qi_k

        if verbose:
            errors_list.append(err)

        if err <= tolerance:
            if verbose:
                print(f"Converged to {tolerance=} after {i=} iterations.")
            break

    if i == steps:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")

    baseline = Qi.mean()
    Vi = np.log(
        np.multiply(prior_policy, np.exp(
            beta * (Qi.reshape((env.nS, env.nA))- baseline))).sum(axis=1)
    ) / beta
    Vi += baseline

    policy = np.multiply(prior_policy, np.exp(
        beta * (Qi.reshape((env.nS, env.nA)) - Vi.reshape(-1,1))))
    pi = policy

    Qi = np.array(Qi).reshape((env.nS, env.nA))

    Vi = np.array(Vi).reshape((env.nS, 1))

    if savename is not None:
        # TODO: allow for arbitrary directory to be made...
        foldername, filename = savename.split('/')
        exists = os.path.exists(foldername)
        if not exists:
            # Create a new directory because it does not exist
            os.makedirs(foldername)

        with open(savename, 'wb+') as f:
            pickle.dump((Qi, Vi, pi), f)

    if verbose:
        return Qi, Vi, pi, qs, errors_list
    else:
        return Qi, Vi, pi


def get_mdp_generator(env, transition_dynamics, policy):
    td_coo = transition_dynamics.tocoo()

    rows, cols, data = [], [], []
    for s_j, col, prob in zip(td_coo.row, td_coo.col, td_coo.data):
        for a_j in range(env.nA):
            row = s_j * env.nA + a_j
            rows.append(row)
            cols.append(col)
            data.append(prob * policy[s_j, a_j])

    nrow = ncol = env.nS * env.nA
    shape = (nrow, ncol)
    mdp_generator = csr_matrix((data, (rows, cols)), shape=shape)

    return mdp_generator

class ModifiedFrozenLake(DiscreteEnv):
    """Customized version of gym environment Frozen Lake"""

    def __init__(
            self, desc=None, map_name="4x4", slippery=0, n_action=4,
            cyclic_mode=True, never_done=True,
            goal_attractor=0.,
            max_reward=0., min_reward=-1.5, step_penalization=1.,
            render_mode=None):

        self.render_mode = render_mode
        goal_attractor = float(goal_attractor)

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min_reward, max_reward)

        if n_action == 2:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = None
        elif n_action == 3:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = 2
        elif n_action in [4, 5]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 4
        elif n_action in [8, 9]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_leftdown = 4
            a_downright = 5
            a_rightup = 6
            a_upleft = 7
            a_stay = 8

        else:
            raise NotImplementedError(f'n_action:{n_action}')

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        isd = np.array(desc == b'S').astype('float64').ravel()
        if isd.sum() == 0:
            isd = np.array(desc == b'F').astype('float64').ravel()
        isd /= isd.sum()
        self.isd = isd

        transition_dynamics = {s : {a : [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == a_left:
                col = max(col - 1, 0)
            elif action == a_down:
                row = min(row + 1, nrow - 1)
            elif action == a_right:
                col = min(col + 1, ncol - 1)
            elif action == a_up:
                row = max(row - 1, 0)
            elif action == a_stay:
                pass
            elif action == a_leftdown:
                col = max(col - 1, 0)
                row = min(row + 1, nrow - 1)
            elif action == a_downright:
                row = min(row + 1, nrow - 1)
                col = min(col + 1, ncol - 1)
            elif action == a_rightup:
                col = min(col + 1, ncol - 1)
                row = max(row - 1, 0)
            elif action == a_upleft:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
        
            else:
                raise ValueError("Invalid action provided")
            return (row, col)

        def compute_transition_dynamics(action_set, action_intended):

            restart = letter in b'H' and cyclic_mode

            diagonal_mode = n_action in [8, 9]

            for action_executed in action_set:
                prob = 1. / (len(action_set) + slippery)
                prob = (slippery + 1) * prob if action_executed == action_intended else prob

                if not restart:
                    newrow, newcol = inc(row, col, action_executed)
                    newletter = desc[newrow, newcol]
                    newstate = to_s(newrow, newcol)

                    if letter == b'G':
                        newletter = letter
                        newstate = state

                    wall_hit = newletter == b'W'
                    if wall_hit:
                        newletter = letter
                        newstate = state
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'
                    ate_candy = letter == b'C'
                    step_nail = letter == b'N'

                    is_diagonal_step = diagonal_mode and action_executed in [4, 5, 6, 7]
                    diagonal_adjust = 1.4 if is_diagonal_step else 1.

                    rew = 0.
                    rew -= step_penalization * (1. - is_in_goal) * diagonal_adjust
                    rew -= step_nail * step_penalization / 2.
                    rew += ate_candy * step_penalization / 2.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward

                    done = is_in_goal and not never_done
                    if is_in_goal:
                        p = prob * goal_attractor
                        if p > 0:
                            sat_li.append((p, newstate, rew, done))
                        for ini_state, start_prob in enumerate(isd):
                            p = start_prob * prob * (1 - goal_attractor)
                            if p > 0.0:
                                sat_li.append((p, ini_state, rew, done))
                    else:
                        sat_li.append((prob, newstate, rew, done))
                else:
                    done = False
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'

                    rew = 0.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward

                    for ini_state, start_prob in enumerate(isd):
                        if start_prob > 0.0:
                            sat_li.append((start_prob * prob, ini_state, rew, done))

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)

                for action_intended in all_actions:
                    sat_li = transition_dynamics[state][action_intended]
                    letter = desc[row, col]

                    if slippery != 0:
                        if action_intended == a_left:
                            action_set = set([a_left, a_down, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_down:
                            action_set = set([a_left, a_down, a_right])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_right:
                            action_set = set([a_down, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_up:
                            action_set = set([a_left, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_stay:
                            action_set = set([a_stay])
                        else:
                            raise ValueError(f"encountered undefined action: {action_intended}")

                    else:
                        action_set = set([action_intended])

                    compute_transition_dynamics(action_set, action_intended)

        self.nS = n_state
        self.nA = n_action
        
        super(ModifiedFrozenLake, self).__init__(n_state, n_action, transition_dynamics, isd)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        transitions = self.P[self.s][int(action)]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        prob, state, reward, terminated = transitions[i]
        self.s = state
        self.lastaction = action
        state = np.array([self.s])
        if self.render_mode == 'human':
            self.render()
        #TODO: update with time limit somehow
        truncated = False
        return (state, reward, terminated, truncated, {"prob": prob})

    def reset(self):
        super(ModifiedFrozenLake, self).reset()
        return np.array([self.s]) , {}

    def render(self):
        if self.render_mode is None:
            return
        
        outfile = StringIO() if self.render_mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if self.render_mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        else:
            return None


MAPS = {
    "hallway1": [
        "SFFFFFG", #H at end?
    ],
    "hallway2": [
        "SFFFFFFFFFG",
    ],
    "hallway3": [
        "SFFFFFFFFFFFFFFG",
    ],
    "2x9ridge": [
        "FFFFFFFFF",
        "FSFHHHFGF"
    ],
    "3x2uturn": [
        "SF",
        "HF",
        "GF",
    ],
    "3x3uturn": [
        "SFF",
        "HHF",
        "GFF",
    ],
    "3x9ridge": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FSFHFHFGF"
    ],
    "5x4uturn": [
        "SFFF",
        "FFFF",
        "HHFF",
        "FFFF",
        "GFFF",
    ],
    "3x4uturn": [
        "SFFF",
        "HHHF",
        "GFFF",
    ],
    "3x5uturn": [
        "SFFFF",
        "HHHHF",
        "GFFFF",
    ],
    "3x6uturn": [
        "SFFFFF",
        "HHHHHF",
        "GFFFFF",
    ],
    "3x7uturn": [
        "SFFFFFF",
        "HHHHHHF",
        "GFFFFFF",
    ],
    "3x12ridge": [
        "FFFHHHHHHFFF",
        "FSFFFFFFFFGF",
        "FFFHHHHHHFFF"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFC"
    ],
    "4x4empty": [
        "FFFF",
        "FSFF",
        "FFGF",
        "FFFF"
    ],
    "5x5empty": [
        "FFFFF",
        "FSFFF",
        "FFFFF",
        "FFFGF",
        "FFFFF"
    ],
    "5x12ridge": [
        "FFFHHHHHHFFF",
        "FFFFFFFFFFFF",
        "FSFFFFFFFFGF",
        "FFFFFFFFFFFF",
        "FFFHHHHHHFFF"
    ],
    "6x6empty": [
        "FFFFFF",
        "FSFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFGF",
        "FFFFFF"
    ],
    "7x7wall": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFWWWFF",
        "FFFFFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x7holes": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFHHHFF",
        "FFFFFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x7wall-mod": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFWWWFF",
        "FFFCFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x8wall": [
        "FFFFFFFF",
        "FFFSFFFF",
        "FFFFFFFF",
        "FFWWWWFF",
        "FFFFFFFF",
        "FFFFGFFF",
        "FFFFFFFF"
    ],
    "7x7zigzag": [
        "FFFFFFF",
        "FSFFFFF",
        "WWWWWFF",
        "FFFFFFF",
        "FFWWWWW",
        "FFFFFCF",
        "FFFFFFF"
    ],
    "8x8empty": [
        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF"
    ],
    "9x9empty": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFSFFFFF",
        "FFFFFFFFF",
        "FFFFFGFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF"
    ],
    "9x9wall": [
        "FFFFFFFFF",
        "FFFFSFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFWWWWWFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFGFFFF",
        "FFFFFFFFF"
    ],
    "9x9zigzag": [
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ],
    "9x9zigzag2h": [
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ],
    "8x8zigzag": [
        "FFFFFFFF",
        "FSFFFFFF",
        "WWWWWFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFWWWWW",
        "FFFFFFGF",
        "FFFFFFFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "8x8a_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF",
        "FFFFWWFF",
        "FFFWFFFF",
        "WWWFFFFF",
        "FFFFFWFF",
        "FFFWWWFF",
        "FFWFFFFF",
        "GFWFFFFF",
    ],
    "8x8b_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF",
        "FFFFWWFF",
        "HHHWFFFF",
        "WWWFFFFF",
        "FFFFFWFF",
        "FFFWWWFF",
        "FFWFFFFF",
        "GFWHHHHH",
    ],
    "5x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "10x10empty": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "9x9channel": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFSFFFFFF",
        "FFWHFHWFF",
        "FFWHFHWFF",
        "FFWHFHWFF",
        "FFFFFFGFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
    ],
    "10x10channel": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFFFFFF",
        "FFFFFFFFFF",
        "FFWHFFHWFF",
        "FFWHFFHWFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "8x8candy": [

        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFCFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF",
    ],
    "10x10candy": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFCFFFFFFF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2-nails": [
        "FFFFFFFFFF",
        "FFFFNFNFNF",
        "FFSFFCFCFF",
        "FFFFNFNFNF",
        "FFFFFFFCFF",
        "FFCFFFNFNF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "11x11gradient": [
        "22222223432",
        "21112234543",
        "21012345654",
        "21112234543",
        "22222123432",
        "22221S12322",
        "22222122222",
        "33333333333",
        "44444444444",
        "55555555555",
        "66666666666",
    ],
    "11x11gradient-x2": [
        "98489444444",
        "84248445754",
        "42024447974",
        "84244445754",
        "98444244444",
        "44442S24444",
        "44444244444",
        "55555555555",
        "66666666666",
        "77777777777",
        "88888888888",
    ],
    "11x11zigzag": [
        "FFFFFFFFFFF",
        "FSFFFFFFFFF",
        "FFFFFFFFFFF",
        "WWWWWWWWFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFWWWWWWWW",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "11x11empty": [
        "FFFFFFFFFFF",
        "FSFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "11x11wall": [
        "FFFFFFFFFFF",
        "FFFFFSFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFWWWWWWWFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFGFFFFF",
        "FFFFFFFFFFF",
    ],
    "11x11dzigzag": [
        "FFFFFFFWFFF",
        "FSFFFFWFFFF",
        "FFFFFWFFFFF",
        "FFFFWFFFFFF",
        "FFFWFFFFFFF",
        "FFWFFFFFWFF",
        "FFFFFFFWFFF",
        "FFFFFFWFFFF",
        "FFFFFWFFFFF",
        "FFFFWFFFFCF",
        "FFFWFFFFFFF",
    ],
    "5x11ridgex2": [
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
        "FSFHHHHHFGF",
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
    ],
    "7x11ridgex4": [
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
        "FSFHHHHHFGF",
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
    ],
    "7x11uturn": [
        "FFFFFFFFFFF",
        "FFFFFFFFFSF",
        "FFFFFFFFFFF",
        "FFFFWWWWWWW",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "7x7hot_uturn": [
        "FFFFFFF",
        "FFFFFSF",
        "FFFFFFF",
        "FFFHWWW",
        "FFFFFFF",
        "FFFFFGF",
        "FFFFFFF",
    ],
    "9x9ridgex4": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFHHHFFF",
        "FFFFFFFFF",
        "FSFHHHFGF",
        "FFFFFFFFF",
        "FFFHHHFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
    ],
    "15x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "15x15zigzag": [
        "FFFFFFFFFFFFFFF",
        "FSFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "WWWWWWWWWWWFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFWWWWWWWWWWW",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGF",
        "FFFFFFFFFFFFFFF",
    ],
    "16x16empty": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candy": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candyx2": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFCFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFCFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16bigS" : [
        "WWWWWWWWWWWWWWWW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFSFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFWWWWWWWWWWW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WWWWWWWWWWWFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFGFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WWWWWWWWWWWWWWWW",
    ],
    "5x17empty": [
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
    ],
    "5x24empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "7x32empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFSFFFFFFFFFFFFFFFFFFFFFFGFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "15x45": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "17x17center": [
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFGFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
    ],
    "5x15zigzag": [
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
    ],
    "8x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "11x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "23x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'WWWWWWWWWWWWFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFWWWWWWWWWWWW',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'WWWWWWWWWWWWFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFWWWWWWWWWWWW',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "15x15mixed": [
        'FFFFFFFSFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'WWWWFFFWWWWWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFWWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFGFFFFFFF',
    ],
    "11x11mixed": [
        'FFFFFSFFFFF',
        'FFFFFWFFFFF',
        'FFFFWWFFFFF',
        'FFFFFWFFFFF',
        'FFFFFWFFFFF',
        'WWWFFWWWWFF',
        'FFFFFWFFFFF',
        'FFFFFWFFFFF',
        'FFFWWWFFFFF',
        'FFFFFWFFFFF',
        'FFFFFGFFFFF',
    ],
    "9x9mixed": [
        'FFFWFWFFF',
        'FFFFSFFFF',
        'FFFWFWFFF',
        'FFFWWWFFF',
        'WWFFWWWWF',
        'FFFFWFFFF',
        'FFWWWFFFF',
        'FFFFGFFFF',
        'FFFFWFFFF',
    ],
    "9x15asymmetric": [
        'FFFFFFWFWFFFFFF',
        'FWWFWFFSFFWWWWF',
        'FWFWFWWFWWWWWWF',
        'FFFFFFFWFFFFFFF',
        'FWFWFWFWFWWWWWW',
        'WFWFWWFWFWWWWWW',
        'FFFFFFFWFFFFFFF',
        'FWWWWWWWWWWWWWF',
        'FFFFFFGFFFFFFFF',
    ],
    "10x11asymmetric": [
        'FFFFWFWFFFF',
        'FFWFFSFFWWF',
        'FWFWWFWWWWF',
        'FFFFFWFFFFF',
        'FWFWFWFWWWW',
        'WFWFFWFWWWW',
        'FFFFFWFFFFF',
        'FWFWFWWWWWF',
        'FFWFWWWWWWF',
        'FFFFGFFFFFF',
    ],
    "9x10asymmetric": [
        'FWFWWFWWWW',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWFWWWWWW',
    ],
    "9x10asymmetric-00": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWFWWWWWW',
    ],
    "9x10asymmetric-01": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-02": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWWWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-03": [
        'WWWWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-04": [
        'WWFWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-05": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'WFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-09": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WWWFWWWWWW',
    ],
    "9x10asymmetric-20": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWWWWWWWW',
    ],
    "9x10asymmetric-21": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'FFWWWWWWWW',
    ],
    "9x10asymmetric-22": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-23": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-24": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-25": [
        'WWFWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-26": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-27": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-28": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'WFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-29": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'WFWWFFFWWF',
        'WFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "empty-quad" :[
        'FGFWFFFFF',
        'FFFWFFFGF',
        'FFFWFFFFF',
        'FFFWFWWWW',
        'FFFFSFFFF',
        'WWWWFWFFF',
        'FFFFFWFGF',
        'FFFGFWFFF',
        'FFFFFWFFF',
    ],
    "Todorov3A":[
        'FFFFF',
        'FFWFF',
        'FWWWF',
        'FFWFF',
        'FFFFG',
    ],
    "Todorov3B":[
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFWWWFF',
        'FFFFFFFFFFFWFFF',
        'FFFFFFFWFFFWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFWWWWWFFFFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFWWWFFFF',
        'FFFFFFFFWFFFFFF',
        'FFFFFFFFFFFFFFF',
    ],
    "Tiomkin2":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWFWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    "Tiomkin2wider":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWFWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    "Tiomkin2zigzag":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    '20x20burger': [
        'WWWWWWWWWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFWWWWWWFFFFFFW',
        'WFFFFFWFFFFFFWFFFFFW',
        'WFFFFWFFFFFFFFWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFSFFFFFFFFFFFFFFGFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWW',
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] not in '#H':
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]
