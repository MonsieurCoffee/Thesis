from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.optimizers import Adam

def create_dqn(learning_rate, input_shape, num_actions, conv_units=32, dense_units=128):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),
        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),
        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),

        Flatten(),

        Dense(dense_units, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(num_actions, activation='linear') # Output Dense Layer = 36 Q(s,a) Values
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate, epsilon=1e-4), loss='mse')
    return model