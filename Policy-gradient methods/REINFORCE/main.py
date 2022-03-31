import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
from gym.spaces import Box
from REINFORCE import *
from logger import logger


# ======================== PARAMETERS ===============================

params = {
    'seed': 3,
    'gpu' : False,
    'env_name': 'LunarLanderContinuous-v2',
    'render': False,
    'n_episodes': 1001,
    'print_info': True,
    'log_info':1000,
    'baseline': True, # if true execute the baseline update rule otherwise the standar one.
    'gamma': 0.99,
    'std': 1,
    'n_hidden_layers': 2,
    'hidden_size': 64,
    'lr': 1e-4,
    'model_summary': False
}

# ===================================================================


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


    ######################## TRAINING ###########################

    mean_reward = deque(maxlen=100)
    gamma = params['gamma']
    std = params['std'] # continuous agent's std dev
    logger = logger(params['env_name'], params['seed'], 'REINFORCE')

    for episode in range(params['n_episodes']):
        ep_reward, steps = 0, 0  
        state = env.reset()
        if params['render']: env.render()

        while True:
            if agent.continuous: 
                action, _ = agent.select_action(state, std)
            else: 
                action = agent.select_action(state, std)

            new_state, reward, done, _ = env.step(action)

            agent.buffer.store(state, action, reward)

            ep_reward += reward
            steps += 1
            if done: break  

            state = new_state

        # after each episode we perform update
        agent.update(gamma, steps, std, params['baseline'])
        mean_reward.append(ep_reward)
        logger.write(episode, ep_reward, np.mean(mean_reward))

        if episode % params['log_info'] == 0: 
            logger.save_model(agent.model, episode, np.mean(mean_reward))

        print(f'Ep: {episode}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')