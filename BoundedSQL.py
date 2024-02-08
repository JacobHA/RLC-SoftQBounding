import time
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
                 clip_target: bool = True,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.clip_target = clip_target
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

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        # return self.env.action_space.sample()
        kl = 0
        return self.online_softqs.choose_action(state), kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        return self.online_softqs.choose_action(state, greedy=True)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch

        # Calculate the current softq values (feedforward):
        curr_softq = torch.cat([online_softq(states).squeeze().gather(1, actions.long())
                               for online_softq in self.online_softqs], dim=1)

        with torch.no_grad():
            online_softq_next = torch.stack([softq(next_states)
                                            for softq in self.online_softqs], dim=0)
            online_curr_softq = torch.stack([softq(states).gather(1, actions)
                                            for softq in self.online_softqs], dim=0)

            # since pi0 is same for all, just do exp(ref_softq) and sum over actions:
            # TODO: should this go outside no grad? Also, is it worth defining a log_prior value?
            
            online_curr_softq = online_curr_softq.squeeze(-1)

            target_next_softqs = [target_softq(next_states)
                                 for target_softq in self.target_softqs]
            target_next_softqs = torch.stack(target_next_softqs, dim=0)

            if self.clip_target:
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

                # lb = torch.max(lb, torch.ones_like(lb)*0)
                # ub = torch.min(ub, torch.ones_like(ub)*1/(1-self.gamma))

                # Recast the bounds to be same shape as target_next_softqs:
                lb = lb.repeat(self.num_nets, 1, self.nA)
                ub = ub.repeat(self.num_nets, 1, self.nA)

                target_next_softqs = torch.clamp(target_next_softqs, min=lb, max=ub)

                # Count number of clips by comparing old and new target:
                num_clips = (old_target != target_next_softqs).sum().item()

                # Average magnitude of bound violations:
                avg_violation = (old_target - target_next_softqs).abs().mean().item()

                # Log these values:
                self.logger.record("train/target_clipped", num_clips)
                self.logger.record("train/avg_violation", avg_violation)

                # Log the average upper bound, lower bound, and target:
                for name, vals in zip(['lb', 'ub', 'target_q'],
                                       [lb, ub, target_next_softqs]):
                    self.logger.record(f"train/{name}_mean", vals.mean().item())


            #TODO: put aggregation here
            next_vs = 1/self.beta * (torch.logsumexp(
                self.beta * target_next_softqs, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = self.aggregator_fn(next_vs, dim=0)

            next_v = next_v.reshape(-1, 1)
            assert next_v.shape == dones.shape
            next_v = next_v * (1-dones)  # + self.theta * dones

            # "Backup" eigenvector equation:
            expected_curr_softq = rewards + self.gamma * next_v
            expected_curr_softq = expected_curr_softq.squeeze(1)

        # Calculate the softq ("critic") loss:
        loss = 0.5*sum(self.loss_fn(softq, expected_curr_softq)
                       for softq in curr_softq.T)
        # log the loss:
        self.logger.record("train/loss", loss.item())
        return loss

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_softqs.polyak(self.online_softqs, self.tau)


