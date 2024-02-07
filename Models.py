import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from stable_baselines3.common.utils import polyak_update, zip_strict
from stable_baselines3.common.preprocessing import preprocess_obs
from gymnasium import spaces
import gymnasium as gym
from torch.optim.lr_scheduler import StepLR, MultiplicativeLR, LinearLR, ExponentialLR, LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import is_tabular

def is_image_space_simple(observation_space, is_vector_env=False):
    if is_vector_env:
        return isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 4
    return isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3
NORMALIZE_IMG = False



class EmptyScheduler(LRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

str_to_scheduler = {
    "step": (StepLR, {'step_size': 100_000, 'gamma': 0.5}),
    # "MultiplicativeLR": (MultiplicativeLR, ()), 
    "linear": (LinearLR, {"start_factor":1./3, "end_factor":1.0, "last_epoch":-1}), 
    "exponential": (ExponentialLR, {'gamma': 0.9999}), 
    "none": (EmptyScheduler, {"last_epoch":-1})
}

class Optimizers():
    def __init__(self, list_of_optimizers: list, scheduler_str: str = 'none'):
        self.optimizers = list_of_optimizers
        scheduler_str = scheduler_str.lower()
        scheduler, params = str_to_scheduler[scheduler_str]
        
        self.schedulers = [scheduler(opt, **params) for opt in self.optimizers]

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt, scheduler in zip(self.optimizers, self.schedulers):
            opt.step()
            scheduler.step()

    def get_lr(self):
        return self.schedulers[0].get_lr()[0]

class TargetNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets
    def __len__(self):
        return len(self.nets)
    def __iter__(self):
        return iter(self.nets)

    def load_state_dicts(self, list_of_state_dicts):
        """
        Load state dictionaries into target networks.

        Args:
            list_of_state_dicts (list): A list of state dictionaries to load into the target networks.

        Raises:
            ValueError: If the number of state dictionaries does not match the number of target networks.
        """
        if len(list_of_state_dicts) != len(self):
            raise ValueError("Number of state dictionaries does not match the number of target networks.")
        
        for online_net_dict, target_net in zip(list_of_state_dicts, self):
            
            target_net.load_state_dict(online_net_dict)

    def polyak(self, online_nets, tau):
        """
        Perform a Polyak (exponential moving average) update for target networks.

        Args:
            online_nets (list): A list of online networks whose parameters will be used for the update.
            tau (float): The update rate, typically between 0 and 1.

        Raises:
            ValueError: If the number of online networks does not match the number of target networks.
        """
        if len(online_nets) != len(self.nets):
            raise ValueError("Number of online networks does not match the number of target networks.")

        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for new_params, target_params in zip(online_nets.parameters(), self.parameters()):
                # for new_param, target_param in zip_strict(new_params, target_params):
                #     target_param.data.mul_(tau).add_(new_param.data, alpha=1.0-tau)
                #TODO: 
                polyak_update(new_params, target_params, tau)

    def parameters(self):
        """
        Get the parameters of all target networks.

        Returns:
            list: A list of network parameters for each target network.
        """
        return [net.parameters() for net in self.nets]


class OnlineNets():
    """
    A utility class for managing online networks in reinforcement learning.

    Args:
        list_of_nets (list): A list of online networks.
    """
    def __init__(self, list_of_nets, aggregator_fn, is_vector_env=False):
        self.nets = list_of_nets
        self.nA = list_of_nets[0].nA
        self.device = list_of_nets[0].device
        self.aggregator_fn = aggregator_fn
        self.is_vector_env = is_vector_env

    def __len__(self):
        return len(self.nets)
    
    def __iter__(self):
        return iter(self.nets)

    def choose_action(self, state, greedy=False, prior=None):
        raise NotImplementedError

    def parameters(self):
        return [net.parameters() for net in self]

    def clip_grad_norm(self, max_grad_norm):
        for net in self:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)




class OnlineSoftQNets(OnlineNets):
    def __init__(self, list_of_nets, aggregator_fn, beta, is_vector_env=False):
        super().__init__(list_of_nets, aggregator_fn, is_vector_env)
        self.beta = beta
           
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            q_as = torch.stack([net.forward(state) for net in self], dim=1)
            q_as = q_as.squeeze(0)
            q_a = self.aggregator_fn(q_as, dim=0)


            if greedy:
                action = torch.argmax(q_a).cpu().numpy()
            else:
                # pi propto e^beta Q:
                # first subtract a baseline from q_a:
                q_a = q_a - (torch.max(q_a) + torch.min(q_a))/2
                clamped_exp = torch.clamp(self.beta * q_a, min=-20, max=20)
                pi = prior * torch.exp(clamped_exp)
                pi = pi / torch.sum(pi)
                a = Categorical(pi).sample()
                action = a.cpu().numpy()
        return action

    
class SoftQNet(torch.nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(SoftQNet, self).__init__()
        self.env = env
        self.nA = env.action_space.n
        self.is_tabular = is_tabular(env)
        self.device = device
        # Start with an empty model:
        model = nn.Sequential()

        self.nS = env.observation_space.shape
        if self.is_tabular:
            nS = env.observation_space.n

        else:
            nS = self.nS[0]
        input_dim = nS
        model.extend(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, self.nA),    
        ))

        model.to(self.device)
        self.model = model
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)  # Convert to PyTorch tensor
        
        # x = x.detach()
        x = preprocess_obs(x, self.env.observation_space, normalize_images=NORMALIZE_IMG)
        assert x.dtype == torch.float32, "Input must be a float tensor."

        if self.is_tabular:
            # Single state
            if x.shape[0] == self.nS:
                x = x.unsqueeze(0)
            else: 
                x = x.squeeze(1)
                pass
        else:
            if len(x.shape) > len(self.nS):
                # in batch mode:
                pass
            else:
                # is a single state
                if isinstance(x.shape, torch.Size):
                    if x.shape == self.nS:
                        x = x.unsqueeze(0)
                else:
                    if (x.shape == self.nS).all():
                        x = x.unsqueeze(0)

        x = self.model(x)
        return x