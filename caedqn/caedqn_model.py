# caedqn/caedqn_model.py
import tensorflow as tf
from tensorflow.keras import layers, Model #type: ignore

def create_caedqn_model(learning_rate, input_shape, num_actions, conv_units=32, dense_units=128):
    """
    Build a DQN model that outputs both Q-values and the embedding vector
    from the penultimate dense layer (used for CAE exploration bonus).

    Returns:
        model: Keras Model with two outputs:
               - 'q_values': (batch, num_actions)
               - 'embedding': (batch, dense_units)
    """
    state_input = layers.Input(shape=input_shape, name='state')

    # CNN feature extractor
    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(state_input)
    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)

    # First dense layer
    x = layers.Dense(dense_units, activation='relu')(x)

    # Embedding layer (penultimate)
    embedding = layers.Dense(dense_units, activation='relu', name='embedding')(x)

    # Q‑values output
    q_values = layers.Dense(num_actions, activation='linear', name='q_values')(embedding)

    model = Model(inputs=state_input, outputs=[q_values, embedding])

    # Compile: only q_values has a loss; embedding is ignored
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-4),
        loss={'q_values': 'mse'}
    )
    return model