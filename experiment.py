import gymnasium
from BoundedSQL import SoftQAgent
import wandb

from tabular import ModifiedFrozenLake
from gymnasium.wrappers import TimeLimit

sql_froz = {
    'batch_size': 32,
    'beta': 5,
    'gamma': 0.98,
    'hidden_dim': 64,
    'learning_rate': 0.0023,
    'learning_starts': 1000,#0.02*50_000,
    'target_update_interval': 100,
    'tau': 1,
    'train_freq': 1,
    'gradient_steps': 1,
}

sql_cpole = {
    'batch_size': 64,
    'beta': 1,
    'gamma': 0.98,
    'hidden_dim': 64,
    'learning_rate': 0.01,
    'learning_starts': 1000,#0.02*50_000,
    'target_update_interval': 100,
    'tau': 0.95,
    'train_freq': 9,
    'gradient_steps': 9,
}

sql_lunar = {
    'batch_size': 128,
    'beta': 10,
    'gamma': 0.98,
    'hidden_dim': 128,
    'learning_rate': 0.003,
    'learning_starts': 0.023*500_000,
    'target_update_interval': 1000,
    'tau': 0.92,
    'train_freq': 5,
    'gradient_steps': 5,
}

sql_mcar = {
    'batch_size': 128,
    'beta': 0.7,
    'gamma': 0.99,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'learning_starts': 0.09*100_000,
    'target_update_interval': 100,
    'tau': 0.97,
    'gradient_steps': 2,
    'train_freq': 2,
}

sql_acro = {
    'batch_size': 128,
    'beta': 2.6,
    'gamma': 0.99,
    'hidden_dim': 64,
    'learning_rate': 0.0066,
    'learning_starts': 0.1*50_000,
    'target_update_interval': 100,
    'tau': 0.92,
    'train_freq': 9,
    'gradient_steps': 9,
}



def main(config=None):
    map_name = '7x7zigzag'
    env_id='FrozenLake-v1'
    env = ModifiedFrozenLake(map_name=map_name,cyclic_mode=False,slippery=0)
    env = TimeLimit(env, max_episode_steps=1000)
    # env_id = TimeLimit(env_id, max_episode_steps=100)

    # env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    env_id = 'LunarLander-v2'
    # env_id = 'ALE/Pong-v5'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'
    env = gymnasium.make(env_id)
    
    id_to_params = {
        'FrozenLake-v1': sql_froz,
        'CartPole-v1': sql_cpole,
        'Acrobot-v1': sql_acro,
        'LunarLander-v2': sql_lunar,
    }

    if not isinstance(env_id, str):
        env_str = map_name
    else:
        env_str = env_id

    
    with wandb.init(project='clipping', entity='jacobhadamczyk', sync_tensorboard=True) as run:
        cfg = run.config
        config = cfg.as_dict()

        clip_method = 'soft'

        # config = id_to_params[env_id]
        default_params = {
            'beta': 5,
            'gamma': 0.98,
            'learning_starts': 0,#1000,
            # 'learning_rate': 0.12,
            # 'perceptron_model': True,
            # 'target_update_interval': 10,
            # 'batch_size': 512,
            # 'soft_weight': 0.0000058,
        }
        wandb.log({'clip_method': clip_method, 'env_id': env_str})#, 'pretrain': pretrain})
        agent = SoftQAgent(env, **default_params, **config,
                            device='cpu', log_interval=1000,
                            tensorboard_log='pong', num_nets=1, 
                            render=False, 
                            clip_method=clip_method)
        
        # Measure the time it takes to learn:
        agent.learn(total_timesteps=300_000)
        wandb.finish()


if __name__ == '__main__':
    # for _ in range(5):
    #     main()
    full_sweep_id='jacobhadamczyk/clipping/jq7ib9su'
    wandb.agent(full_sweep_id, function=main, count=500)
    # main()