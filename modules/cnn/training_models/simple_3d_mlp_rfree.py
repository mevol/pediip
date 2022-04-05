"""Testing with training_pipeline"""

from typing import Tuple

#from keras import Sequential, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

from modules.cnn.training_models.training_pipeline_rfree import pipeline_from_command_line

def create_3D_cnn_model(input_shape: Tuple[int, int, int, int]):
    model = Sequential()

    
    model.add(Dense(512, activation="relu", input_shape=input_shape))
    model.add(Dense(1, activation="linear"))

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
