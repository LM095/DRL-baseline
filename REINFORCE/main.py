import argparse
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import yaml
from gym.spaces import Box
from REINFORCE import *


# ===================== PARAMETERS =========================
params = {
    'seed': 3,
    'gpu' : False,
    'env_name': 'LunarLander-v2',
    'n_episodes': 1001,
    'print_info': True,
    'log_info':50,
    'baseline': False, # if true execute the baseline update rule otherwise the standar one.
    'gamma': 0.99,
    'std': 1,
    'n_hidden_layers': 2,
    'hidden_size': 64,
    'lr': 1e-4,
    'model_summary': False
}

# ================================================================


os.environ['PYTHONHASHSEED'] = str(params['seed'])

if __name__ == '__main__':
    env = gym.make(params['env_name'])
    env.seed(params['seed'])

    # check discrete or continuous agent based on the env chosen
    if (isinstance(env.action_space, Box)):
        if params['print_info']: print('Reinforce_Continuous')
        agent = REINFORCE(env, params, continuous=True)
    else:
        if params['print_info']: print('Reinforce_Discrete')
        agent = REINFORCE(env, params, continuous=False)

    agent.train(params)