import argparse
import yaml
from BoundedSQL import SoftQAgent
import wandb
from utils import sample_wandb_hyperparams


env_to_steps = {
    'MountainCar-v0': 500_000,
    'CartPole-v1': 10_000,
    'Acrobot-v1': 2_000,
}

int_hparams = {'train_freq', 'gradient_steps'}

def main(sweep_config=None, project=None, ft_params=None, fine_tune=False, log_dir='tf_logs', hparam_dir='hparams', device='auto'):
    total_timesteps = env_to_steps[env_id]
    runs_per_hparam = 3
    avg_auc = 0
    unique_id = wandb.util.generate_id()

    # sample the hyperparameters from wandb locally
    wandb_kwargs = {"project": project}
    if sweep_config:
        sweep_config["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(sweep_config["parameters"], int_hparams=int_hparams)
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = sampled_params
    elif ft_params:
        full_config = {}
        default_params = yaml.safe_load(open(f'{hparam_dir}/{env_id}-none.yaml'))
        full_config.update(default_params)
        if ft_params is not None:
            # Overwrite the default params:
            full_config.update(ft_params)
        wandb_kwargs['config'] = full_config

    # run runs_per_hparam for each hyperparameter set
    for i in range(runs_per_hparam):
        unique_id = unique_id[:-1] + f"{i}"
        with wandb.init(sync_tensorboard=True, id=unique_id, group="ft" if fine_tune else None, **wandb_kwargs) as run:
            cfg = run.config
            print(run.id)
            config = cfg.as_dict()
            # save the first config if sampled from wandb to use in the following runs_per_hparam
            wandb_kwargs['config'] = config

            default_params = yaml.safe_load(open(f'{hparam_dir}/{env_id}-none.yaml'))
            full_config = {}
            full_config.update(default_params)
            if ft_params is not None:
                # Overwrite the default params:
                full_config.update(ft_params)
                # Log the new params (for group / filtering):
                wandb.log(ft_params)
            else:
                full_config.update(config)

            wandb.log({'env_id': env_id})

            # cast sampled params to int if they are in int_hparams
            for k in int_hparams:
                full_config[k] = int(full_config[k])

            agent = SoftQAgent(env_id, **full_config,
                                device=device, log_interval=500,
                                tensorboard_log=log_dir, num_nets=1,
                                render=False,)

            # Measure the time it takes to learn:
            agent.learn(total_timesteps=total_timesteps)
            avg_auc += agent.eval_auc
            wandb.log({'avg_auc': avg_auc / runs_per_hparam})


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=100)
    args.add_argument('--project', type=str, default='clipping')
    args.add_argument('--do_sweep', type=bool, default=False)
    args.add_argument('--env_id', type=str, default='Acrobot-v1')
    args.add_argument('--clip_method', type=str, default='none')
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--hparam-dir', type=str, default='hparams')
    args = args.parse_args()
    env_id = args.env_id
    device = args.device

    if args.do_sweep:
        # Run a hyperparameter sweep with w&b:
        print("Running a sweep on wandb...")
        sweep_cfg = yaml.safe_load(open('sweeps/classic-sweep.yaml'))
        for i in range(args.count):
            main(sweep_cfg, project=args.project, device=device, fine_tune=False, log_dir='sw_logs')

    else:
        # Run finetuned hyperparameters:
        print("Running finetuned hyperparameters...")
        clip_method = args.clip_method
        assert clip_method in ['none', 'hard', 'soft-huber', 'soft-linear', 'soft-square'], \
            "Invalid clip method. Choose from: none, hard, soft-huber, soft-linear, soft-square\
            or add a hyperparameter file in the \'hparams\' folder."
        acrobot = yaml.safe_load(open(f'{args.hparam_dir}/{env_id}-{clip_method}.yaml'))
        for i in range(args.count):
            main(None, project=args.project, fine_tune=True, ft_params=acrobot, log_dir='ft_logs', hparam_dir=args.hparam_dir, device=device)

if __name__ == '__main__':
    # for _ in range(20):
    main()
