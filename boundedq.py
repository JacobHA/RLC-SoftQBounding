import torch as th
from torch.nn import functional as F
from torch.optim import Adam
import numpy as np

from stable_baselines3.dqn import DQN


class BoundedDQN(DQN):
    """
    DQN variant with bounded Q updates:
    upper bound: r(s,a) + gamma * E_{s'} v^pi(s') + delta + gamma * max delta / (1-gamma)
    lower bound: r(s,a) + gamma * E_{s'} v^pi(s') + delta + gamma * min delta / (1-gamma)
    delta(s,a) = r(s,a) + gamma * E_{s'} {max|min}_a' Q^pi(s',a') - Q^pi(s,a)
    assuming V^pi(s) = max_a Q^pi(s,a), and that the environment is deterministic
    """
    def __init__(self, *args, **kwargs):
        self.clip_target = kwargs.pop('clip_target', True)
        self.clip_gradient = kwargs.pop('clip_gradient', True)
        super(BoundedDQN, self).__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_value = self.q_net_target(replay_data.next_observations)
                # TODO: Possibly clip here?

                # Follow greedy policy: use the one with the highest value
                max_q_value, idx_max = next_q_value.max(dim=1, keepdim=True)
                # Avoid potential broadcast issue
                max_q_value = max_q_value.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + \
                    (1 - replay_data.dones) * self.gamma * max_q_value
                # clip target to be between min and max values
                # get probabilities of taking the min and max actions
                curr_q_values = self.policy.q_net(replay_data.observations)
                next_q_values = self.policy.q_net(
                    replay_data.next_observations)
                # curr_v_min, _ = curr_q_values.min(dim=1, keepdim=True)
                curr_v_max, _ = curr_q_values.max(dim=1, keepdim=True)
                # next_v_min, _ = next_q_values.min(dim=1, keepdim=True)
                next_v_max, _ = next_q_values.max(dim=1, keepdim=True)
                delta = replay_data.rewards + \
                    (1 - replay_data.dones) * \
                    self.gamma * next_v_max - curr_v_max
                delta = delta.reshape(-1, 1)
                delta_min = th.min(delta)
                delta_max = th.max(delta)
                clipped_target_q_values = th.max(th.min(
                    target_q_values,
                    replay_data.rewards + delta + delta_max *
                    self.gamma / (1 - self.gamma)
                ), replay_data.rewards + delta + delta_min * self.gamma / (1 - self.gamma),
                )
                if self.clip_target:
                    target_q_values = clipped_target_q_values
                # Count how many times the target was clipped
                num_clips = (clipped_target_q_values ==
                             replay_data.rewards + delta + delta_max * self.gamma / (1 - self.gamma)).sum().item()
                percent_clips = num_clips / batch_size
                self.logger.record("train/target_clipped", percent_clips)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # TODO: clip at the online network?

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.clip_gradient:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


class BoundedDQNv2(DQN):
    """
    DQN variant with bounded Q updates:
    upper bound: r(s,a) + gamma * E_{s'} v^pi(s') + delta + gamma * max delta / (1-gamma)
    lower bound: r(s,a) + gamma * E_{s'} v^pi(s') + delta + gamma * min delta / (1-gamma)
    delta(s,a) = r(s,a) + gamma * E_{s'} {max|min}_a' Q^pi(s',a') - Q^pi(s,a)
    assuming V^pi(s) = min_a Q^pi(s,a) for a conservative update,
    and that the environment is deterministic
    """
    def __init__(self, *args, **kwargs):
        self.clip_target = kwargs.pop('clip_target', True)
        self.clip_gradient = kwargs.pop('clip_gradient', True)
        super(BoundedDQNv2, self).__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_value = self.q_net_target(replay_data.next_observations)
                min_q_value, idx_min = next_q_value.min(dim=1, keepdim=True)
                # Follow greedy policy: use the one with the highest value
                max_q_value, idx_max = next_q_value.max(dim=1, keepdim=True)
                # Avoid potential broadcast issue
                max_q_value = max_q_value.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + \
                    (1 - replay_data.dones) * self.gamma * max_q_value
                # clip target to be between min and max values
                # get probabilities of taking the min and max actions
                curr_q_values = self.policy.q_net(replay_data.observations)
                next_q_values = self.policy.q_net(
                    replay_data.next_observations)
                curr_v, _ = curr_q_values.min(dim=1, keepdim=True)
                next_v, _ = next_q_values.min(dim=1, keepdim=True)
                delta = replay_data.rewards + \
                    (1 - replay_data.dones) * \
                    self.gamma * next_v - curr_v
                delta = delta.reshape(-1, 1)
                delta_min = th.min(delta)
                delta_max = th.max(delta)
                clipped_target_q_values = th.max(th.min(
                    target_q_values,
                    replay_data.rewards + delta + delta_max *
                    self.gamma / (1 - self.gamma)
                ), replay_data.rewards + delta + delta_min * self.gamma / (1 - self.gamma),
                )
                if self.clip_target:
                    target_q_values = clipped_target_q_values
                # Count how many times the target was clipped
                num_clips = (clipped_target_q_values ==
                             replay_data.rewards + delta + delta_max * self.gamma / (1 - self.gamma)).sum().item()
                percent_clips = num_clips / batch_size
                self.logger.record("train/target_clipped", percent_clips)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            if self.clip_gradient:
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))