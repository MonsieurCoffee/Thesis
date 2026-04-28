import random
import numpy as np
from collections import deque
from dqn_model import create_dqn
import tensorflow as tf

MEMORY_SIZE = 300_000           # Replay Buffer Size
MEMORY_MIN = 10_000             # Minimum Transitions Needed to Start Training
BATCH_SIZE = 128                # Transition Batching Size for Training
LEARNING_RATE = 0.001           # Learning Rate
DISCOUNT_FACTOR = 0.9           # Gamma
EPSILON = 0.95                  # Epsilon Greedy Policy
EPSILON_DECAY = 0.99995         # Epsilon Decay
EPSILON_MIN = 0.02              # Minimum Decay
UPDATE_TARGET_EVERY = 1000      # Copy Main Network to Target Network
CONV_UNITS = 32                 # Convolutional Filters
DENSE_UNITS = 128               # Dense Neurons
TRAIN_EVERY = 4                 # Update Weights & Biases Every 4 Environment Steps

# Default Model Name
MODEL_NAME = f'conv{CONV_UNITS}x3_dense{DENSE_UNITS}x2_y{DISCOUNT_FACTOR}'

class DQNAgent:
    # Initialize Agent
    def __init__(self, env, model_name=MODEL_NAME, seed=None, conv_units=CONV_UNITS, dense_units=DENSE_UNITS):
        self.env = env
        self.model_name = model_name

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.discount = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.batch_size = BATCH_SIZE
        self.train_every = TRAIN_EVERY
        self.train_step_counter = 0

        # Initialize Board Dimensions
        input_shape = self.env.state_im.shape
        num_actions = self.env.n_tiles

        # Model Initialization (Main & Target)
        self.model = create_dqn(self.learning_rate, input_shape, num_actions, conv_units, dense_units)          # Online Network (Main)
        self.target_model = create_dqn(self.learning_rate, input_shape, num_actions, conv_units, dense_units)   # Target Network
        self.target_model.set_weights(self.model.get_weights())                                                 # Set Target = Online Network

        # Initialize Experience Replay Buffer
        self.replay_memory = deque(maxlen=MEMORY_SIZE)  # FIFO
        self.target_update_counter = 0

    # Action Selection Logic
    def get_action(self, state):
        # Masking & Action Space
        unknown_channel = state[:, :, 9]                                # Extract Channel 9 (Unknown)
        flat_unknown = unknown_channel.reshape(-1)                      # Flatten 2D Array to 1D
        unsolved = [i for i, u in enumerate(flat_unknown) if u == 1.0]  # Valid Moves

        # Epsilon Greedy Policy
        if np.random.random() < self.epsilon:
            action = np.random.choice(unsolved)
        else:
            input_batch = np.expand_dims(state, axis=0)
            q_values = self.model.predict_on_batch(input_batch)[0]
            mask = (flat_unknown == 0)
            q_values[mask] = np.min(q_values)
            action = np.argmax(q_values)
        return action

    # Store Transitions to Replay Buffer
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Training
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        # Estimate Best Future Q-Values {argmaxQ(s',a')}
        future_qs = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_qs, axis=1)

        # Target Network (TD Target)
        targets = rewards + self.discount * max_future_q * (1.0 - tf.cast(dones, tf.float32))

        # Forward Pass Inside Gradient Tape to Compute Loss
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)                    # Prediction
            actions_one_hot = tf.one_hot(actions, depth=self.env.n_tiles)   # Board Representation (One-Hot)
            selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)  # Q Value of Chosen Action
            loss = tf.reduce_mean(tf.square(targets - selected_q))          # MSE

        # Compute Gradients & Update Weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # Training Preparation
    def train(self, done):
        # Train Every 4 Environment Steps
        self.train_step_counter += 1
        if self.train_step_counter % self.train_every != 0:
            return
        
        # Wait Until Replay Buffer >= 10k
        if len(self.replay_memory) < MEMORY_MIN:
            return

        # Sample 128 Random Transitions
        batch = random.sample(self.replay_memory, self.batch_size)

        # Arrays to Store 128 Random Transitions Batch (Batch Storage)
        state_shape = self.env.state_im.shape
        X = np.empty((self.batch_size, *state_shape), dtype=np.float32)
        actions_arr = np.empty(self.batch_size, dtype=np.int32)
        rewards_arr = np.empty(self.batch_size, dtype=np.float32)
        next_states_arr = np.empty((self.batch_size, *state_shape), dtype=np.float32)
        dones_arr = np.empty(self.batch_size, dtype=np.bool_)

        # Fill Arrays With Actual Data From Replay Buffer
        for i, (state, action, reward, next_state, terminal) in enumerate(batch):
            X[i] = state
            actions_arr[i] = action
            rewards_arr[i] = reward
            next_states_arr[i] = next_state
            dones_arr[i] = terminal

        # Give Batch Storage to _train_step to Start Learning
        self._train_step(
            tf.convert_to_tensor(X, dtype=tf.float32),
            tf.convert_to_tensor(actions_arr, dtype=tf.int32),
            tf.convert_to_tensor(rewards_arr, dtype=tf.float32),
            tf.convert_to_tensor(next_states_arr, dtype=tf.float32),
            tf.convert_to_tensor(dones_arr, dtype=tf.bool)
        )

        # Copy Main Network to Target Network
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # Epsilon Decay Mechanism
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)