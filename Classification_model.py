from keras.api.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, Activation, Input
from keras.api.models import Sequential
from keras.api.initializers import glorot_uniform
from keras.api.optimizers import Adam
from keras.api.regularizers import l2

# def get_classification_model(im_size):
#     model = Sequential([
#         Conv2D(32, (2, 2), activation="relu", input_shape=(im_size, im_size, 3)),
#         MaxPooling2D(2, 2),
#
#         Flatten(),
#         Dense(128, activation="relu"),
#         Dense(15, activation="softmax")
#     ])
#
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#     return model


def get_classification_model(input_shape=(45, 45, 1)):
    regularizer = l2(0.01)
    model = Sequential()
    model.add(Input(shape=(45, 45, 1)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                     kernel_initializer=glorot_uniform(seed=0),
                     name='conv1', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act1'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                     kernel_initializer=glorot_uniform(seed=0),
                     name='conv2', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                     kernel_initializer=glorot_uniform(seed=0),
                     name='conv3', activity_regularizer=regularizer))
    model.add(Activation(activation='relu', name='act3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dense(84, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Dense(15, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc3'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model