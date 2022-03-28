import numpy as np
import random

class Buffer():
    def __init__(self):
        self.buffer = []

    def store(self, state, action, reward, new_state, done):
        self.buffer.append([state, action, reward, new_state, done])

    def sample(self, batch_size):

        # we randomly choose between the buffer size and the batch size
        samples = random.sample(self.buffer, batch_size)

        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples]) 
        new_state = np.array([sample[3] for sample in samples]) 
        done = np.array([sample[4] for sample in samples])    

        return states, actions, rewards, new_state, done

    def clear(self):
        self.buffer.clear()

    def size(self): 
        return len(self.buffer)
    