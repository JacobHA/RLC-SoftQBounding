import numpy as np
import torch
from BaseAgent import BaseAgent
from Models import SoftQNet, OnlineSoftQNets, Optimizers, TargetNets
from utils import logger_at_folder
from bound_utils import bounds

class SoftQAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 clip_method: str = None,
                 pretrain: bool = False,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.clip_method = clip_method

        self.total_clips = 0
        self.pretrain = pretrain
        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_softqs = OnlineSoftQNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device)
                                              for _ in range(self.num_nets)],
                                            beta=self.beta,
                                            aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_softqs

        self.target_softqs = TargetNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_softqs.load_state_dicts(
            [softq.state_dict() for softq in self.online_softqs])
        # Make (all) softqs learnable:
        opts = [torch.optim.Adam(softq.parameters(), lr=self.learning_rate)
                for softq in self.online_softqs]
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        return self.online_softqs.choose_action(state)

    def evaluation_policy(self, state: np.ndarray) -> int:
        return self.online_softqs.choose_action(state, greedy=True)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch
        # rewards -= (1-self.gamma) * 32

        with torch.no_grad():
            online_softq_next = torch.stack([softq(next_states)
                                            for softq in self.online_softqs], dim=0)
            online_curr_softq = torch.stack([softq(states).gather(1, actions)
                                            for softq in self.online_softqs], dim=0)

            online_curr_softq = online_curr_softq.squeeze(-1)

            target_next_softqs = [target_softq(next_states)
                                 for target_softq in self.target_softqs]
            target_next_softqs = torch.stack(target_next_softqs, dim=0)

            old_target = target_next_softqs
            target_curr_softqs = torch.stack([softq(states)
                                            for softq in self.target_softqs], dim=0)
            target_lb, target_ub = bounds(self.beta, self.gamma, rewards, dones, actions, target_next_softqs, target_curr_softqs)

            online_curr_softqs = torch.stack([softq(states)
                                            for softq in self.online_softqs], dim=0)
            online_lb, online_ub = bounds(self.beta, self.gamma, rewards, dones, actions, online_softq_next, online_curr_softqs)

            # Take best bounds:
            lb = torch.max(online_lb, target_lb)
            ub = torch.min(online_ub, target_ub)

            # Count number of clips by comparing old and new target:
            num_clips = (old_target != target_next_softqs).sum().item()
            self.total_clips += num_clips
            self.logger.record("train/total_clips", self.total_clips)

            # Average magnitude of bound violations:
            avg_violation = (old_target - target_next_softqs).abs().mean().item()

            # Log these values:
            self.logger.record("train/target_clipped", num_clips)
            self.logger.record("train/avg_violation", avg_violation)

            # Log the average upper bound, lower bound, and target:
            for name, vals in zip(['lb', 'ub', 'target_q'],
                                    [lb, ub, target_next_softqs]):
                self.logger.record(f"train/{name}_mean", vals.mean().item())


            # aggregate the target next softqs:
            target_next_softq = self.aggregator_fn(target_next_softqs, dim=0)
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * target_next_softq, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))

            next_v = next_v.reshape(-1, 1)
            # next_v = 32 + next_v ---> "deviation"/perturbation

            # "Backup" eigenvector equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)
            expected_curr_softq = expected_curr_softq.squeeze(1)
            # clip the expected curr soft q:
            if 'hard' in self.clip_method:
                expected_curr_softq = torch.clamp(expected_curr_softq, min=lb.squeeze(), max=ub.squeeze())

        curr_softq = torch.stack([softq(states).squeeze().gather(1, actions.long())
                        for softq in self.online_softqs], dim=0)

        # clip the online curr soft q, then gather the actions:
        lb = lb.unsqueeze(0).repeat(self.num_nets, 1, 1)
        ub = ub.unsqueeze(0).repeat(self.num_nets, 1, 1)
        
        clipped_curr_softq = torch.clamp(curr_softq, min=lb, max=ub)
        # clipped_curr_softq = clipped_curr_softq.squeeze(2)
        if 'online' in self.clip_method:
            curr_softq = clipped_curr_softq
     
        # num_nets, batch_size, 1 (leftover from actions)
        curr_softq = curr_softq.squeeze(2)

        # log the mean online q :

        # 32 * perturbation
        self.logger.record("train/online_q_mean", curr_softq.mean().item())
        # Calculate the softq ("critic") loss:
        loss = 0.5*sum(self.loss_fn(softq, expected_curr_softq)
                       for softq in curr_softq)
        if 'soft' in self.clip_method:
            # add the magnitude of bound violations to the loss:
            clip_loss = ((clipped_curr_softq - curr_softq)**2).sum()
            # log the clip loss:
            self.logger.record("train/clip_loss", clip_loss.detach().item())
            loss += 0.05 * clip_loss
        # log the loss:
        self.logger.record("train/loss", loss.item())
        return loss

    def pretrain_descent(self, gradient_steps: int, batch_size: int):
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size)
            states, actions, next_states, dones, rewards = batch
            softq_next = torch.stack([softq(next_states)
                                            for softq in self.online_softqs], dim=0)
            softq_curr = torch.stack([softq(states)
                                            for softq in self.online_softqs], dim=0)
            lb, ub = bounds(self.beta, self.gamma, rewards, dones, actions, softq_next, softq_curr)
            # aim to minimize the difference between the upper and lower bounds:
            loss = sum(self.loss_fn(q.mean(dim=1).unsqueeze(1), ub) for q in softq_curr)

            self.optimizers.zero_grad()
            loss.backward()
            self.optimizers.step()
            self.logger.record("train/pre-loss", loss.item())

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_softqs.polyak(self.online_softqs, self.tau)
