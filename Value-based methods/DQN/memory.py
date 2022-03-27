import numpy as np

class Buffer():
    def __init__(self):
        self.buffer = []

    def store(self, state, action, reward, new_state, done):
        self.buffer.append([state, action, reward, new_state, done])

    def sample(self):
        states = np.array([sample[0] for sample in self.buffer])
        actions = np.array([sample[1] for sample in self.buffer])
        rewards = np.array([sample[2] for sample in self.buffer]) 
        new_state = np.array([sample[3] for sample in self.buffer]) 
        done = np.array([sample[4] for sample in self.buffer])    

        return states, actions, rewards, new_state, done

    def clear(self):
        self.buffer.clear()

    def size(self): 
        return len(self.buffer)
    