import gymnasium
from BoundedSQL_learned import SoftQAgent
import wandb

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
    'learning_starts': 100,#0.02*50_000,
    'target_update_interval': 100,
    'tau': 0.95,
    'train_freq': 9,
    'gradient_steps': 9,
}

sql_lunar = {
    'batch_size': 32,
    'beta': 7.07,
    'gamma': 0.99,
    'hidden_dim': 128,
    'learning_rate': 0.00014,
    'learning_starts': 5_000,
    'target_update_interval': 10,
    'tau': 0.92,
    'train_freq': 5,
    'gradient_steps': 49,
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
    'batch_size': 64,
    'beta': 4.5,
    'gamma': 0.99,
    'hidden_dim': 64,
    'learning_rate': 0.0005,
    'learning_starts': 0.0,
    'target_update_interval': 10,
    'tau': 0.92,
    'train_freq': 2,
    'gradient_steps': 20,
}



def main(config=None):
    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    env_id = 'Acrobot-v1'
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

    env_str = env_id
    
    with wandb.init(project='clipping', entity='jacobhadamczyk', sync_tensorboard=True) as run:
        cfg = run.config
        config = cfg.as_dict()

        clip_method = 'naivereset-learned-soft'
        # clip_method = 'hard'

        default_params = id_to_params[env_id]
        default_params['soft_weight'] = 0.01
        # default_params['learning_rate'] = 1e-3
        # default_params['batch_size'] = 1200
        
        wandb.log({'clip_method': clip_method, 'env_id': env_str})#, 'pretrain': pretrain})
        agent = SoftQAgent(env, **default_params, **config,
                            device='auto', log_interval=1000,
                            tensorboard_log='pong', num_nets=1, 
                            render=False,
                            clip_method=clip_method)
        
        # Measure the time it takes to learn:
        agent.learn(total_timesteps=150_000)
        wandb.finish()


if __name__ == '__main__':
    for _ in range(20):
        main()
    # full_sweep_id='jacobhadamczyk/clipping/p76w2p4l'
    # wandb.agent(full_sweep_id, function=main, count=500)
    # # main()