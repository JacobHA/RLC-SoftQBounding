import wandb
from tabular_soln import main

runs_per_hparam = 3
env = '7x7zigzag'
gamma = 0.98

def _wandb_func(clip, naive, lr):
    rwds = 0
    print(lr)
    for _ in range(runs_per_hparam):
        rwds, gap = main(env_str=env, clip=clip, lr=lr, gamma=gamma, oracle=False, naive=naive, save=False)
        rwds += rwds
    wandb.log({'avg_reward': rwds / runs_per_hparam})


options = {
    'none': {'clip': False, 'naive': False},
    'hard': {'clip': True, 'naive': False},
    'naive': {'clip': True, 'naive': True},
}

# OPTION = 'hard'
import random
OPTION = random.choice(list(options.keys()))
full_sweep_id = 'jacobhadamczyk/clipping/s6u20w12'

def wandb_func(config=None):
    with wandb.init(project='clipping', entity='jacobhadamczyk') as run:
        cfg = run.config
        config = cfg.as_dict()

        clip = options[OPTION]['clip']
        naive = options[OPTION]['naive']
        lr = config['learning_rate']
        _wandb_func(clip, naive, lr)
        wandb.log({'clip_method': clip, 'naive': naive})
        

if __name__ == '__main__':
    wandb.agent(full_sweep_id, function=wandb_func, count=5)