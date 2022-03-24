import numpy as np

class Buffer():
    def __init__(self):
        self.buffer = []

    def store(self, state, action, reward):
        self.buffer.append([state, action, reward])

    def sample(self):
        states = np.array([sample[0] for sample in self.buffer])
        actions = np.array([sample[1] for sample in self.buffer])
        rewards = np.array([sample[2] for sample in self.buffer])        
        return states, actions, rewards

    def clear(self):
        self.buffer.clear()
    