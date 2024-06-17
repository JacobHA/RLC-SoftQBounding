import gymnasium
from BoundedSQL_learned import SoftQAgent
import wandb
from gymnasium.wrappers import TimeLimit

ENTITY = 'your_username_here'

sql_cpole = {
    'batch_size': 64,
    'beta': 1,
    'gamma': 0.98,
    'hidden_dim': 64,
    'learning_rate': 0.01,
    'learning_starts': 0,
    'target_update_interval': 100,
    'tau': 0.95,
    'train_freq': 9,
    'gradient_steps': 9,
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

id_to_timesteps = {
    'MountainCar-v0': 500_000,
    'CartPole-v1': 10_000,
    'Acrobot-v1': 5_000,
}


def main(config=None):
    env_id = 'CartPole-v1'
    # env_id = 'Acrobot-v1'
    # env_id = 'MountainCar-v0'

    env = gymnasium.make(env_id)
    
    id_to_params = {
        'MountainCar-v0': sql_mcar,
        'CartPole-v1': sql_cpole,
        'Acrobot-v1': sql_acro,
    }

   
    with wandb.init(project='clipping', entity=ENTITY, sync_tensorboard=True) as run:
        cfg = run.config
        config = cfg.as_dict()

        clip_method = 'hard'

        default_params = id_to_params[env_id]

        wandb.log({'clip_method': clip_method, 'env_id': env_id})
        agent = SoftQAgent(env, **default_params, **config,
                            device='auto', log_interval=1000,
                            tensorboard_log='logs', num_nets=1, 
                            render=False,
                            clip_method=clip_method)
        
        # Measure the time it takes to learn:
        agent.learn(total_timesteps=id_to_timesteps[env_id])
        wandb.finish()


if __name__ == '__main__':
    for _ in range(20):
        main()
