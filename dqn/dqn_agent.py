# encoder_dqn/dqn/dqn_agent.py
import random
import numpy as np
from collections import deque
from dqn_model import create_dqn
import tensorflow as tf # type: ignore

# ========== Hyperparameters ==========
MEMORY_SIZE = 300_000
MEMORY_MIN = 10_000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
LEARNING_DECAY = 1.0          # no decay
DISCOUNT_FACTOR = 0.9
EPSILON = 0.95
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.02
UPDATE_TARGET_EVERY = 1000    # steps
CONV_UNITS = 32
DENSE_UNITS = 128
TRAIN_EVERY = 4


MODEL_NAME = f'conv{CONV_UNITS}x3_dense{DENSE_UNITS}x2_y{DISCOUNT_FACTOR}'


class DQNAgent:
    def __init__(self, env, model_name=MODEL_NAME, seed=None,
                 conv_units=CONV_UNITS, dense_units=DENSE_UNITS):
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

        input_shape = self.env.state_im.shape   # (h, w, 10)
        num_actions = self.env.n_tiles

        self.model = create_dqn(self.learning_rate, input_shape, num_actions,
                                conv_units, dense_units)
        self.target_model = create_dqn(self.learning_rate, input_shape, num_actions,
                                       conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        self.target_update_counter = 0

    def get_action(self, state):
        """
        state shape: (h, w, 10) – one‑hot encoding.
        Unsolved tiles have channel 9 == 1.0
        """
        # Flatten unknown channel
        unknown_channel = state[:, :, 9]          # shape (h, w)
        flat_unknown = unknown_channel.reshape(-1) # (n_tiles,)
        unsolved = [i for i, u in enumerate(flat_unknown) if u == 1.0]

        if np.random.random() < self.epsilon:
            action = np.random.choice(unsolved)
        else:
            # Expand dims to batch size 1
            input_batch = np.expand_dims(state, axis=0)   # (1, h, w, 10)
            q_values = self.model.predict_on_batch(input_batch)[0]
            # Mask revealed tiles (unknown == 0) to minimum Q
            mask = (flat_unknown == 0)
            q_values[mask] = np.min(q_values)
            action = np.argmax(q_values)
        return action

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """
        TensorFlow‑compiled training step.
        States shape: (batch_size, h, w, 10)
        """
        current_qs = self.model(states, training=False)
        future_qs = self.target_model(next_states, training=False)
        max_future_q = tf.reduce_max(future_qs, axis=1)

        targets = rewards + self.discount * max_future_q * (1.0 - tf.cast(dones, tf.float32))

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            actions_one_hot = tf.one_hot(actions, depth=self.env.n_tiles)
            selected_q = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            loss = tf.reduce_mean(tf.square(targets - selected_q))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, done):
        self.train_step_counter += 1
        if self.train_step_counter % self.train_every != 0:
            return

        if len(self.replay_memory) < MEMORY_MIN:
            return

        batch = random.sample(self.replay_memory, self.batch_size)

        # Pre‑allocate arrays
        state_shape = self.env.state_im.shape
        X = np.empty((self.batch_size, *state_shape), dtype=np.float32)
        actions_arr = np.empty(self.batch_size, dtype=np.int32)
        rewards_arr = np.empty(self.batch_size, dtype=np.float32)
        next_states_arr = np.empty((self.batch_size, *state_shape), dtype=np.float32)
        dones_arr = np.empty(self.batch_size, dtype=np.bool_)

        for i, (state, action, reward, next_state, terminal) in enumerate(batch):
            X[i] = state
            actions_arr[i] = action
            rewards_arr[i] = reward
            next_states_arr[i] = next_state
            dones_arr[i] = terminal

        # Compiled training step
        self._train_step(
            tf.convert_to_tensor(X, dtype=tf.float32),
            tf.convert_to_tensor(actions_arr, dtype=tf.int32),
            tf.convert_to_tensor(rewards_arr, dtype=tf.float32),
            tf.convert_to_tensor(next_states_arr, dtype=tf.float32),
            tf.convert_to_tensor(dones_arr, dtype=tf.bool)
        )

        # Update target network based on steps (not episodes)
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)