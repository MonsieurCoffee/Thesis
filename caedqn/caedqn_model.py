import tensorflow as tf
from tensorflow.keras import layers, Model #type: ignore

def create_caedqn_model(learning_rate, input_shape, num_actions, conv_units=32, dense_units=128):
    state_input = layers.Input(shape=input_shape, name='state')

    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(state_input)
    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(conv_units, (3, 3), activation='relu', padding='same')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(dense_units, activation='relu')(x)
    embedding = layers.Dense(dense_units, activation='relu', name='embedding')(x)           # Embedding Layer
    q_values = layers.Dense(num_actions, activation='linear', name='q_values')(embedding)   # Output Dense Layer = 36 Q(s,a) Values

    model = Model(inputs=state_input, outputs=[q_values, embedding])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-4),
        loss={'q_values': 'mse'}
    )
    return model