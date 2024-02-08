from BoundedSQL import SoftQAgent
import wandb

sql_cpole = {
    'batch_size': 64,
    'beta': 0.1,
    'gamma': 0.98,
    'hidden_dim': 64,
    'learning_rate': 0.01,
    'learning_starts': 0.02*50_000,
    'target_update_interval': 100,
    'tau': 0.95,
    'train_freq': 9,
    'gradient_steps': 9,
}


def main():
    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'ALE/Pong-v5'
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    wandb.init(project='clipping', entity='jacobhadamczyk', sync_tensorboard=True)
    clip = True
    wandb.log({'clip': clip})
    agent = SoftQAgent(env_id, **sql_cpole, device='cuda', log_interval=500,
                 tensorboard_log='pong', num_nets=2, render=False, aggregator='min',
                 scheduler_str='none', clip_target=clip)
    # Measure the time it takes to learn:
    agent.learn(total_timesteps=20_000)
    wandb.finish()


if __name__ == '__main__':
    for _ in range(5):
        main()