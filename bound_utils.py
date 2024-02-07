import torch as th


def net_to_delta(gamma, rewards, dones, next_q_values, curr_q_values):
    next_v_max, _ = next_q_values.max(dim=1, keepdim=True)
    delta = rewards + \
        (1 - dones) * \
        gamma * next_v_max - curr_q_values
    delta = delta.reshape(-1, 1)
    delta_min = th.min(delta)
    delta_max = th.max(delta)
    return delta, delta_min, delta_max

def bounds(beta, gamma, rewards, dones, next_q_values, curr_q_values):
    lb = -th.ones_like(rewards) * float('inf')
    ub = th.ones_like(rewards) * float('inf')
    for (next_q_value, curr_q_value) in zip(next_q_values, curr_q_values):
        delta, delta_min, delta_max = net_to_delta(gamma, rewards, dones, next_q_value, curr_q_value)
        # V = next_q_value.max(dim=1)[0].reshape(-1, 1)
        nA = next_q_value.shape[-1]
        device = next_q_value.device
        V = 1/beta * th.logsumexp(next_q_value * beta, dim=1, keepdim=True) - th.log(th.tensor([nA], device=device))
        lb = rewards + gamma * (V + delta_min / (1 - gamma) )
        ub = rewards + gamma * (V + delta_max / (1 - gamma) )
        # Successively take the better bound:
        lb = th.max(lb, lb)
        ub = th.min(ub, ub)
    
    return lb, ub
