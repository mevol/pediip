"""Testing with training_pipeline"""

from typing import Tuple

#from keras import Sequential, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.layers import GlobalAveragePooling3D, BatchNormalization

from modules.cnn.training_models.training_pipeline_volume_classification import pipeline_from_command_line

def create_3D_cnn_model(input_shape: Tuple[int, int, int, int]):
    model = Sequential()
    # filter from 64 to 8 --> 09/04/2022
    model.add(Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               activation="relu", input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling3D())
    model.add(Dense(16, activation="relu", kernel_regularizer = regularizers.l2(1e-4)))#was 128
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))#was tanh

    print(model.output_shape)


    model.compile(
        loss="binary_crossentropy",
#        optimizer=optimizers.adam(lr=1e-5),
#        optimizer='adam',
#        optimizer=Adam(learning_rate=0.0001),
        optimizer=SGD(learning_rate=0.0001), # was added 20220509
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":

    pipeline_from_command_line(create_3D_cnn_model, rgb=False)
