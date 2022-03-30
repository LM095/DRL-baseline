# DRL-baseline



### Requirements:

**Tensorflow**: [![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)  

**OpenAI Gym**: [![Gym](https://badge.fury.io/py/gym.svg)](https://badge.fury.io/py/gym)


This repo contains a personal implementation of different value-based and policy-gradient Deep Reinforcement Learning (DRL) methods. 
The implemented algorithms are separated into two different folders: 

- **Value-based** methods:
  - [x] DQN [1]
  - [ ] DDQN [2]
  - [ ] Dueling DQN [3] 

- **Policy-gradient** methods:
  - [x] REINFORCE [4]
  - [ ] DDPG [5]
  - [ ] PPO Monte Carlo (PPO_MC) [6]
  - [ ] PPO Temporal Difference (PPO_TD) [6]
  - [ ] Soft Actor-Critic (SAC) [7]
  - [ ] Twin Delayed DDPG (TD3) [8]

All the methods in every subfolder rely on a single algorithm written in Tensorflow2 and Keras. The code is commented in detail to facilitate understanding and replicability in other frameworks such as PyTorch.



### Results

According to the chosen methods, each algorithm was tested on different runs with random seeds over 1000 episodes on two different OpenAI gym environments (continuous and/or discrete cases). The plot shows the average reward of the last 100 episodes and the number of episodes for each method.



| Environment                                 | Plot                                 |
| ------------------------------------------- | ------------------------------------ |
| LunarLander-v2 and LunarLanderContinuous-v2 | ![](https://i.imgur.com/gdJSeNb.png) |
| ![](https://i.imgur.com/6CbaInI.gif)        |                                      |
| Cartpole-v1                                 |                                      |
| ![](https://i.imgur.com/R6yBLwS.gif)        |                                      |





### References

[1] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

[2] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

[3] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

[4] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html)

[5] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

[6] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[7] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

[8] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
