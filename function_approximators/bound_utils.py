import torch as th
from torch.nn import functional as F

def net_to_delta(beta, gamma, rewards, dones, actions, next_q_values, curr_q_values):

    nA = next_q_values.shape[-1]
    device = next_q_values.device
    V = 1/beta * th.logsumexp(next_q_values * beta, dim=1, keepdim=True) - th.log(th.tensor([nA], device=device))

    delta = (rewards + gamma * V * (1 - dones)- curr_q_values) * (1 - dones)

    delta_min = th.min(delta)
    delta_max = th.max(delta)

    return delta, delta_min, delta_max

def bounds(beta, gamma, rewards, dones, actions, next_q_values, curr_q_values):
    with th.no_grad():
        lb = -th.ones_like(rewards) * float('inf')
        ub = th.ones_like(rewards) * float('inf')
        for (next_q_value, curr_q_value) in zip(next_q_values, curr_q_values):
            delta, delta_min, delta_max = net_to_delta(beta, gamma, rewards, dones, actions, next_q_value, curr_q_value)
            nA = next_q_value.shape[-1]
            device = next_q_value.device
            V = 1/beta * (th.logsumexp(next_q_value * beta, dim=1, keepdim=True) - th.log(th.tensor([nA], device=device)))
            # lb = rewards + gamma * (V + delta_min / (1 - gamma) ) * (1 - dones)
            # ub = rewards + gamma * (V + delta_max / (1 - gamma) ) * (1 - dones)
            
            # Successively take the better bound:
            lb = th.max(th.stack([lb, rewards + gamma * (V + delta_min / (1 - gamma) ) * (1 - dones)]), dim=0).values
            ub = th.min(th.stack([ub, rewards + gamma * (V + delta_max / (1 - gamma) ) * (1 - dones)]), dim=0).values
        
        return lb, ub


def calculate_clip_loss(values, 
                        clipped_values, 
                        clip_method='soft-huber', 
                        clip_loss_fn=None, 
                        reduction='mean'):
    
    """Calculates the clip loss based on the chosen clip method and loss function.

    Args:
        clipped_curr_softq (torch.Tensor): The clipped Q-value estimates.
        curr_softq (torch.Tensor): The original Q-value estimates.
        clip_method (str, optional): The clip method to use. Defaults to 'soft-huber'.
        clip_loss_fn (callable, optional): A custom clip loss function. Defaults to None.
        reduction (str, optional): Reduction method for the loss calculation.
            Defaults to 'mean'.

    Returns:
        torch.Tensor: The clip loss value.
    """

    clipped_values = clipped_values.squeeze(2)  # Squeeze dimension 2 if needed
    # diff = clipped_values - values

    if clip_loss_fn is None:
        if clip_method == 'huber':
            clip_loss = F.huber_loss(clipped_values, values, reduction=reduction)
        elif clip_method == 'linear':
            clip_loss = F.l1_loss(clipped_values, values, reduction=reduction)
        elif clip_method == 'square':
            clip_loss = F.mse_loss(clipped_values, values, reduction=reduction)
        else:
            raise ValueError(f"Invalid clip method: {clip_method}")
    else:
        clip_loss = clip_loss_fn(diff)

    return clip_loss