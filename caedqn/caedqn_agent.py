import random
import numpy as np
from collections import deque
from caedqn_model import create_caedqn_model
import tensorflow as tf

MEMORY_SIZE = 300_000           # Replay Buffer Size
MEMORY_MIN = 10_000             # Minimum Transitions Needed to Start Training
BATCH_SIZE = 128                # Transition Batching Size for Training
LEARNING_RATE = 0.001           # Learning Rate
DISCOUNT_FACTOR = 0.9           # Gamma
UPDATE_TARGET_EVERY = 1000      # Copy Main Network to Target Network
CONV_UNITS = 32                 # Convolutional Filters
DENSE_UNITS = 128               # Dense Neurons
TRAIN_EVERY = 4                 # Update Weights & Biases Every 4 Environment Steps

# CAE HYPERPARAMETERS
BETA = 0.1                      # Exploration Coefficient
RIDGE = 1.0                     # Regularisation for Gram Matrix (λ)
RECALC_INV_EVERY = 10000        # Steps Needed Before Full Inverse Recalculation

MODEL_NAME = f'caedqn_conv{CONV_UNITS}x3_dense{DENSE_UNITS}x2'

class CAEDQNAgent:
    # Initialize Agent
    def __init__(self, env, model_name=MODEL_NAME, seed=None, conv_units=CONV_UNITS, dense_units=DENSE_UNITS, beta=BETA, ridge=RIDGE):
        self.env = env
        self.model_name = model_name
        self.beta = beta
        self.ridge = ridge
        self.embedding_dim = dense_units
        self.step_count = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.discount = DISCOUNT_FACTOR
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.train_every = TRAIN_EVERY
        self.train_step_counter = 0

        # Initialize Board Dimensions
        input_shape = self.env.state_im.shape
        num_actions = self.env.n_tiles

        # Model Initialization (Main & Target)
        self.model = create_caedqn_model(self.learning_rate, input_shape, num_actions, conv_units, dense_units)
        self.target_model = create_caedqn_model(self.learning_rate, input_shape, num_actions, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        # Gram Matrix A and Inverse
        self.A = np.eye(self.embedding_dim) * self.ridge        # A = λ·I
        self.A_inv = np.eye(self.embedding_dim) / self.ridge    # A⁻¹ = I/λ

        # Bonus scaling (Welford's Algorithm)
        self.bonus_mean = 0.0      # Running Mean μ
        self.bonus_var = 0.0       # Cumulative Squared Deviation ν²
        self.bonus_count = 0       # Number of Samples N

        # Initialize Experience Replay Buffer
        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        self.target_update_counter = 0

    # Action Selection Logic
    def get_action(self, state):
        # Masking & Action Space
        unknown_channel = state[:, :, 9]            # Extract Channel 9 (Unknown)
        flat_unknown = unknown_channel.reshape(-1)  # Flatten 2D Array to 1D

        state_batch = np.expand_dims(state, axis=0)
        q_values, embedding = self.model.predict_on_batch(state_batch)  # type: ignore
        q_values = q_values[0]
        phi = embedding[0]

        # Compute Raw Bonus
        A_inv_phi = np.dot(self.A_inv, phi)
        raw_bonus = self.beta * np.sqrt(np.dot(phi, A_inv_phi))

        # Running Normalisation (Welford)
        self.bonus_count += 1
        delta = raw_bonus - self.bonus_mean
        self.bonus_mean += delta / self.bonus_count
        delta2 = raw_bonus - self.bonus_mean
        self.bonus_var += delta * delta2
        std = np.sqrt(self.bonus_var / self.bonus_count) if self.bonus_count > 1 else 1.0
        scaled_bonus = raw_bonus / (std + 1e-8)

        # Add Bonus to Q-Values
        augmented_q = q_values + scaled_bonus

        # Mask Revealed Tiles (Set Q to Infinity)
        mask = (flat_unknown == 0)
        augmented_q[mask] = -np.inf

        action = int(np.argmax(augmented_q))

        # Update Gram matrix Once / Step (Sherman-Morrison)
        denom = 1.0 + np.dot(phi, A_inv_phi)
        numerator = np.outer(A_inv_phi, A_inv_phi)
        self.A_inv -= numerator / denom
        self.A += np.outer(phi, phi)

        # Periodic Full Inverse Recalculation
        self.step_count += 1
        if self.step_count % RECALC_INV_EVERY == 0:
            self.A_inv = np.linalg.inv(self.A)

        return action

    # Scaled Bonus
    def compute_bonus(self, state):
        # Fetch Embedding
        state_batch = np.expand_dims(state, axis=0)
        _, embedding = self.model.predict_on_batch(state_batch)  # type: ignore
        phi = embedding[0]

        # Compute Raw UCB Bonus
        A_inv_phi = np.dot(self.A_inv, phi)
        raw_bonus = self.beta * np.sqrt(np.dot(phi, A_inv_phi))

        # Scale Bonus Using Welford's Normalization
        if self.bonus_count > 1:
            std = np.sqrt(self.bonus_var / self.bonus_count)
        else:
            std = 1.0
        scaled_bonus = raw_bonus / (std + 1e-8)
        return float(scaled_bonus)

    # Store Transitions to Replay Buffer
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Training
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        # Get Current Q‑values
        q_values, _ = self.model(states, training=True)

        # Get target Q‑values
        future_qs, _ = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_qs, axis=1)

        # Target Network (TD Target)
        targets = rewards + self.discount * max_future_q * (1.0 - tf.cast(dones, tf.float32))

        # Forward Pass Inside Gradient Tape to Compute Loss
        with tf.GradientTape() as tape:
            q_values, _ = self.model(states, training=True)                 # Prediction
            actions_one_hot = tf.one_hot(actions, depth=self.env.n_tiles)   # Board Representation (One-Hot)
            selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)  # Q Value of Chosen Action
            loss = tf.reduce_mean(tf.square(targets - selected_q))          # MSE

        # Compute Gradients & Update Weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # type: ignore
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