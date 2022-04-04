from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from memory import Buffer




class REINFORCE:
    def __init__(self, env, params, continuous):

        np.random.seed(params['seed'])
        tf.random.set_seed(params['seed'])
        self.env = env
        self.continuous = continuous
        if continuous: self.std = params['std']

        self.model = self.build_model(env, params, continuous, name='model_REINFORCE')
        self.optimizer = Adam()
        self.buffer = Buffer()




    def build_model(self, env, params, continuous, name):
        
        input_size = env.observation_space.shape[0]

        if continuous: 
            action_size = env.action_space.shape[0]
        else: 
            action_size = env.action_space.n

        n_hidden_layers = params['n_hidden_layers']
        hidden_size = params['hidden_size']

        # Build the network with the collected parameters
        state_input = Input(shape=(input_size,), name='input_layer')
        h = state_input

        for i in range(n_hidden_layers):
            h = Dense(hidden_size, activation='relu', name='hidden_' + str(i))(h)
          
        if continuous:
            output = Dense(action_size, activation='tanh', name='output_layer')(h)  
        else:
            output = Dense(action_size, activation='softmax', name='output_layer')(h)
        
        model = Model(inputs=state_input, outputs=output)

        # PNG with the architecture and summary
        if params['model_summary']:
            plot_model(model, to_file=name + '.png', show_shapes=True)    
            model.summary()

        return model


    def select_action(self, state, std=1.0):
       
        if self.continuous:
            mu = self.model(np.array([state])).numpy()[0]
            action = np.random.normal(loc=mu, scale=std**2)    
            return action, mu

        # if we are in the discrete case, passing the state as input to the DNN, we get a softmax probability distribution, and we select as action a random value in the distribution: example, 
        # let's suppose we have 5 action with p=[0.1, 0, 0.3, 0.6, 0], i.e., the probability of selecting the action 0 with this particular state is 0.1, and so on... 
        # if we run: np.random.choice(5,p) we will obtain 2 or 3 since they are the ones with the highest probability.
        probs = self.model(np.array([state])).numpy()[0]
        action = np.random.choice(self.env.action_space.n, p=probs)

        return action



    def update(self, gamma, steps, std=1.0, baseline=False):
        # we take the samples and we compute the cumulative reward in order to perform the update
        # gamma is the discount factor
        # std is used in the Gaussian distribution
        # steps perform during the episode
  

        states, actions, rewards = self.buffer.sample()

        discounted_returns = []
        # for each timestep of the episode
        for t in range(len(rewards)):
            G = 0.0
            # we compute the discounted cumulative reward
            for k, r in enumerate(rewards[t:]):
                G += (gamma**k)*r
            discounted_returns.append(G)
       
        discounted_returns = np.array(discounted_returns)

        # We subtract the mean episode reward as baseline to reduce variance
        if baseline:
            b = np.mean(discounted_returns)   
            discounted_returns -= b

        # Normalize the reward values to reduce variance as introduced in the summary at page 
        discounted_returns -= np.mean(discounted_returns)
        discounted_returns /= np.std(discounted_returns + 1e-7)

        # reshape for update(n° samples, len(reward)) --> G_t (δ if we use baseline)
        G = discounted_returns.reshape(-1, 1)

        if self.continuous:
            self.update_continuous(states, actions, G, std)
        else:
            self.update_discrete(states, actions, G)

        # After the update, we clear the buffer
        self.buffer.clear()

    def update_discrete(self, states, actions, rewards):
       
        # compute the update rule: ∇θJ(θ) ≈ 1/N * ∑ [(∇θ logPπθ(A∣S)) * ((∑ r) - baseline)]
        with tf.GradientTape() as t:

            # Compute Pπθ(a|s) = πθ(A|S)
            # since we are in the discrete case the self.model(states) returns a softmax distribution
            probs = self.model(states)

            # probs initially is a np array of shape (steps preformed during the episode, 4) i.e., for each row (each step in the episode) we have 4 values that
            # represent the probability of taking each action in that step.
            # idxs is used to say at step 0 was taken action x and so on, so idxs is a np.array of shape (steps_preformed, 2) 
            # i.e., for each row (each step in the episode) we have 2 values: the step and the action taken in the env
            idxs = np.array([[i, action] for i, action in enumerate(actions)])

            # action_probs is a np.array of shape (steps_perfomed, 1) i.e., for each steps the probability related to the action taken in the env
            # example with one value: let's suppose we have 
            # probs = np.array[[0.26, 0,29, 0.20, 0.25]] so idxs = np.array([0, 1]) and finally action_probs = np.array[[0.29]]
            action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

            # Take the log(Pπθ(a|s)) = logπθ(A|S)
            log_probs = tf.math.log(action_probs)

            # Compute the actual objective: mean of ( (logπθ(A|S) * G_T (or δ if we use baseline) ) 
            objective = tf.math.multiply(rewards, log_probs)
            # We negate it as we want to max the objective
            objective = -tf.math.reduce_mean(objective)

            # Compute the gradient and update the network
            grads = t.gradient(objective, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_continuous(self, states, actions, rewards, std):

        with tf.GradientTape() as tape:

            # Compute the Gaussian's mu from the network, then Pπθ(ai,t∣si,t)
            mu = self.model(states)
            gauss_num = tf.math.exp(-0.5 * ((actions - mu) / (std))**2)
            gauss_denom = std * tf.sqrt(2 * np.pi)
            gauss_probs = gauss_num / gauss_denom

            # Sum/average/do nothing with the contribution of the n continuous actions
            gauss_probs = tf.math.reduce_mean(gauss_probs, axis=1, keepdims=True)  

            # Take the logPπθ(a∣s)
            log_probs = tf.math.log(gauss_probs)

            # Compute the actual objective 1/N * (log(Pπθ(a|s)) * (∑ r)
            objective = tf.math.multiply(rewards, log_probs)
            # We negate it as we want to max the objective
            objective = -tf.math.reduce_mean(objective)

            # Compute the gradient and update the network
            grads = tape.gradient(objective, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
