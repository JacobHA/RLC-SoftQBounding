import torch as th

def net_to_delta(beta, gamma, rewards, dones, actions, next_q_values, curr_q_values):

    nA = next_q_values.shape[-1]
    device = next_q_values.device
    V = 1/beta * th.logsumexp(next_q_values * beta, dim=1, keepdim=True) - th.log(th.tensor([nA], device=device))

    delta = rewards + gamma * V * (1-dones) - curr_q_values

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
            V = 1/beta * th.logsumexp(next_q_value * beta, dim=1, keepdim=True) - th.log(th.tensor([nA], device=device))
            # lb = rewards + gamma * (V + delta_min / (1 - gamma) ) * (1 - dones)
            # ub = rewards + gamma * (V + delta_max / (1 - gamma) ) * (1 - dones)
            
            # Successively take the better bound:
            lb = th.max(th.stack([lb, rewards + gamma * (V + delta_min / (1 - gamma) ) * (1 - dones)]), dim=0).values
            ub = th.min(th.stack([ub, rewards + gamma * (V + delta_max / (1 - gamma) ) * (1 - dones)]), dim=0).values
        
        return lb, ub
