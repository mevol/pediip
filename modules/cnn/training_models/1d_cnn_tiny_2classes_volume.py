"""Testing with training_pipeline"""

from typing import Tuple

#from keras import Sequential, optimizers
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization

from modules.cnn.training_models.training_pipeline_1D_classification import pipeline_from_command_line

def create_1D_cnn_model(input_shape: Tuple[int, int]):
    model = Sequential()
    # filter from 64 to 8 --> 09/04/2022
    model.add(Conv1D(8, kernel_size=2, strides=1, padding='same',
               activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation="relu", kernel_regularizer = regularizers.l2(1e-4)))#was 128
#    model.add(Dropout(0.3))
 #   model.add(Dense(2, activation="softmax"))#was tanh
    model.add(Dense(1, activation="sigmoid"))

    print(model.output_shape)

    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

    model.compile(
        loss="binary_crossentropy",
#        optimizer=optimizers.adam(lr=1e-5),
#        optimizer='adam',
#        optimizer=Adam(learning_rate=0.0001),
#        optimizer=SGD(learning_rate=0.001), # was added 20220509
        optimizer=Adam(learning_rate=lr_schedule), # was added 20220513
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    pipeline_from_command_line(create_1D_cnn_model, rgb=False)
