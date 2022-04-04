from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from memory import Buffer




class DDQN:
    def __init__(self, env, params):

        np.random.seed(params['seed'])
        tf.random.set_seed(params['seed'])
        self.env = env

        self.model = self.build_model(env, params, name='model_DDQN')

        # in DDQN we use two DNN one for the target and one for the update
        self.model_target = self.build_model(env, params, name='model_target_DDQN')
        # we copy the same weights for both the DNNs
        self.model_target.set_weights(self.model.get_weights())

        self.optimizer = Adam()
        self.buffer = Buffer()


    def build_model(self, env, params, name):
        
        input_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        n_hidden_layers = params['n_hidden_layers']
        hidden_size = params['hidden_size']

        # Build the network with the collected parameters
        state_input = Input(shape=(input_size,), name='input_layer')
        h = state_input

        for i in range(n_hidden_layers):
            h = Dense(hidden_size, activation='relu', name='hidden_' + str(i))(h)
          
        output = Dense(action_size, activation='linear', name='output_layer')(h)
        
        model = Model(inputs=state_input, outputs=output)

        # PNG with the architecture and summary
        if params['model_summary']:
            plot_model(model, to_file=name + '.png', show_shapes=True)    
            model.summary()

        return model


    def select_action(self, state, epsilon):
       
        # we use epsilon greedy strategy to select action to perform in the env
        # if a random value between 0 and 1 is less or equal to epsilon --> continue with exploration, i.e. select a random action
        # otherwise exploit the policy 
        if np.random.uniform() <= epsilon:
            return np.random.randint(0, self.env.action_space.n)
        
        q_values = self.model(np.array([state])).numpy()[0]
        return np.argmax(q_values)



    def update(self, gamma, batch_size):
        # we compute the mse of the temporal difference error given by Q(s,a|θ) and the target y = r + γ max_a' Q(s', a'|θ). 
        # with the help of the target network for stability

        # update the batch_size to avoid errors
        batch_size = min(self.buffer.size(), batch_size)
        states, actions, rewards, new_states, dones = self.buffer.sample(batch_size)

        # we reshape the rewards in order to perform the update
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        
        with tf.GradientTape() as t:
            # Unlike the classic DQN where the target was calculated with the same network of the update, here two different networks are used
            # Compute the target y = r + γ max_ã' Qw(š′,ã′), check dispensa_DRL.pdf
            # Compute the target y = r + γ max_a' Q_tg(s', a'|θ), where a' is computed with model_target
            obs_qvalues_target = self.model_target(new_states)
            
            obs_qvalues = self.model(new_states)
            obs_actions = tf.math.argmax(obs_qvalues, axis=-1).numpy()
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(obs_actions)])

            max_obs_qvalues = tf.expand_dims(tf.gather_nd(obs_qvalues_target, idxs), axis=-1)
            y = rewards + gamma * max_obs_qvalues * dones

            # Compute values Q(s,a|θ)
            qvalues = self.model(states)
            idxs = np.array([[int(i), int(action)] for i, action in enumerate(actions)])
            qvalues = tf.expand_dims(tf.gather_nd(qvalues, idxs), axis=-1)

            # Compute the loss as mse of Q(s, a) - y
            td_errors = tf.math.subtract(qvalues, y)
            td_errors = 0.5 * tf.math.square(td_errors)
            loss = tf.math.reduce_mean(td_errors)

            # Compute the model gradient and update the network
            grad = t.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
