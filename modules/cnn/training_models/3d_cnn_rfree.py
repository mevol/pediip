"""Testing with training_pipeline"""

from typing import Tuple

#from keras import Sequential, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D

from modules.cnn.training_models.training_pipeline_rfree import pipeline_from_command_line

def create_3D_cnn_model(input_shape: Tuple[int, int, int, int]):
    model = Sequential()

    model.add(
        Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(
        Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation="relu"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation="relu"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
#    model.add(Dense(2, activation="softmax"))
    model.add(Dense(2, activation="softmax"))

    print(model.output_shape)


    model.compile(
        loss="categorical_crossentropy",
#        optimizer=optimizers.adam(lr=1e-5),
        optimizer='adam',
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    pipeline_from_command_line(create_3D_cnn_model, rgb=False)
