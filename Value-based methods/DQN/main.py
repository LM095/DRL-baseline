import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
from gym.spaces import Box
from DQN import *
from logger import logger


# ======================== PARAMETERS ===============================

params = {
    'seed': 4321,
    'gpu' : False,
    'env_name': 'LunarLander-v2',
    'render': False,
    'n_episodes': 1001,
    'print_info': True,
    'log_info':1000,
    'update_after': 10,
    #######################
    'gamma': 0.99,
    'epsilon': 1,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.99,
    'buffer_size': 2000,
    'batch_size': 64,
    #######################
    'n_hidden_layers': 2,
    'hidden_size': 64,
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
        agent = DQN(env, params, continuous=True)
    else:
        if params['print_info']: print('Reinforce_Discrete')
        agent = DQN(env, params, continuous=False)


    ######################## TRAINING ###########################

    mean_reward = deque(maxlen=100)
    gamma = params['gamma']
    epsilon = params['epsilon']
    epsilon_min = params['epsilon_min']
    epsilon_decay = params['epsilon_decay']
    batch_size = params['batch_size']
    
    logger = logger(params['env_name'], params['seed'], 'DQN')

    for episode in range(params['n_episodes']):
        ep_reward, steps = 0, 0  
        state = env.reset()
        if params['render']: env.render()

        while True:
            action = agent.select_action(state, epsilon)

            new_state, reward, done, _ = env.step(action)

            agent.buffer.store(state, action, reward, new_state, 1-int(done))

            ep_reward += reward
            steps += 1
            state = new_state

            if episode > params['update_after']: 
                agent.update(gamma, batch_size)

            if done: break  

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        mean_reward.append(ep_reward)
        logger.write(episode, ep_reward, np.mean(mean_reward))

        if episode % params['log_info'] == 0: 
            logger.save_model(agent.model, episode, np.mean(mean_reward))

        print(f'Ep: {episode}, Ep_Rew: {ep_reward}, Mean_Rew: {np.mean(mean_reward)}')