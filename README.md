Welcome to the repository for the [RLC 2024 paper titled "Boosting Soft Q-Learning by Bounding" by Jacob Adamczyk, Volodymyr Makarenko, Stas Tiomkin, and Rahul Kulkarni](https://arxiv.org/abs/2406.18033)
Here you can reproduce the paper's experiments, and try bounded soft Q-learning on your own environments.


We use Gymnasium for RL environments and wandb for logging / hyperparameter tuning.

## Setup

To prepare an environment with conda:

1. setup a conda env
```conda create --name qbounds python=3.10```
2. activate the conda env
```conda activate qbounds```
3. python requirements: 
```pip install -r requirements.txt```

## FA experiments

### reproduce the finetuned results
1. Run the best hparams for an environment    
```python experiments.py --env_id=CartPole-v1```

2. plot the results for the spcific environment    
```python tb_plotter.py --env_id=CartPole-v1```

### re-run the hparam sweep
1. Run the hparam sweep for an environment    
```python experiments.py --env_id=CartPole-v1 --do_sweep=True```
2. extract the best hparams logged with wandb   
```python wandb_best_hparams.py --env_id=CartPole-v1 --entity=your_wandb_entity```

## Tabular experiments

### reproduce the figures
1. Run the lr sweep for each of the methods of interest. It will use 30 mazes contained in the mazes directory.
```python tabular_experiments/tabular_lr_sensitivity.py```
