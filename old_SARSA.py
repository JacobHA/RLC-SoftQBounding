"""Several Q algorithms"""

from icml23.utils.q_table_agents import QTableAgent
from icml23.utils.q_table_agents import EntRegQTableAgent
from icml23.utils.tools import clipper
import numpy as np


class SARSA(QTableAgent):
    """SARSA algorithm"""

    def _update_variables(self, sarsa_experience, done, min_clip, max_clip):
        state, action, reward, next_state, next_action = sarsa_experience

        q_valu = self.q_table[state, action]
        q_next = self.q_table[next_state, next_action]

        q_valu = \
            (1. - self.learning_rate) * q_valu + \
            self.learning_rate * (reward + self.gamma * q_next * (1.0 - done))

        if min_clip is not None and max_clip is not None:
            q_valu, clipped = clipper(
                q_valu, min_clip[state, action], max_clip[state, action])

            if clipped:
                self.clip_counter.append(self.clip_counter[-1] + 1)
            else:
                self.clip_counter.append(self.clip_counter[-1])

        self._update_q_table(state, action, q_valu)

    def _evaluate_policy(self, reg=True, tolerance=1e-8, max_iter=2500):
        """ Returns the current policy's evaluation. """
        pi_value = self.q_table.copy()
        # pi_value += np.max(self.env.reward_range)
        diff = tolerance + 1
        k = 0
        while (diff > tolerance) and (k < max_iter):
            k += 1
            action = np.random.choice(self.action_set)
            state = self.env.reset()
            self.env.state = np.random.choice(self.n_states)
            done = False
            pi_prev_value = pi_value.copy()
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = np.random.choice(
                    self.action_set, p=self.policy[next_state])
                if reg:
                    # Make a choice for next_action given next_state
                    H = -np.log(self.policy[next_state]
                                * self.n_actions) * self.boltzmann_temperature
                    pi_value[state, action] = reward + self.gamma * (
                        (pi_value[next_state] + H).dot(self.policy[next_state]))
                else:
                    pi_value[state, action] = \
                        (reward + self.gamma *
                         pi_value[next_state, next_action])

                state = next_state
                action = next_action
            # else:
            #     pi_value[state, action] = reward
            diff = np.abs(pi_value - pi_prev_value).max()

        return pi_value

    def save(self, path):
        """Save the agent"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        """Load the agent"""
        import pickle
        with open(path, 'rb') as f:
            agent = pickle.load(f)


class EntRegSARSA(EntRegQTableAgent, SARSA):
    """SARSA algorithm with entropy regularization"""

    def _update_variables(self, sarsa_experience, done, min_clip, max_clip):
        state, action, reward, next_state, next_action = sarsa_experience
        beta = 1. / self.boltzmann_temperature

        q_valu = self.q_table[state, action]
        v_next = self.v_vectr[next_state]

        updated_q_valu = (1. - self.learning_rate) * q_valu + \
            self.learning_rate * (reward + self.gamma * v_next)

        if min_clip is not None and max_clip is not None:
            clipped_updated_q_valu, clipped = clipper(
                updated_q_valu, min_clip[state, action], max_clip[state, action])
            if clipped:
                updated_q_valu = clipped_updated_q_valu
                self.clip_counter.append(self.clip_counter[-1] + 1)
            else:
                self.clip_counter.append(self.clip_counter[-1])

        self._update_q_table(state, action, updated_q_valu)
        # Do v clipping here
        # Calculate the clip bounds for the v vector!!!:
        q_s_valu = self.q_table[state]

        delta = (q_s_valu.min() + q_s_valu.max()) / 2.
        if delta in [np.inf, -np.inf]:
            v_valu = delta
        else:
            with np.errstate(over='ignore'):
                v_valu = delta + \
                    np.log(np.exp(beta * (q_s_valu - delta)).sum()
                           * self.prior_poli) / beta

        # self._update_v_vectr(state, v_vectr[state])
        self._update_v_vectr(state, v_valu)

    def save(self, path):
        """Save the agent"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        """Load the agent"""
        import pickle
        with open(path, 'rb') as f:
            agent = pickle.load(f)