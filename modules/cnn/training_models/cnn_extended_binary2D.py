"""Testing with training_pipeline_2d"""

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization #added 20210226
from modules.cnn.training_models.training_pipeline_binary2D import pipeline_from_command_line

def create_2D_cnn_model(input_shape: Tuple[int, int, int]):
    print(1111, input_shape)
    model = Sequential()

    #reduced number of filters 32 --> 16 20210303
    model.add(
        Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same',
               activation="relu", input_shape=input_shape)#kernel 3 --> 5 20210226
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #added layer with 32 filters 20210303
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))#input_shape, padding removed
    #added layer with 32 filters 20210303
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))#input_shape, padding removed
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#strides added 20210303
    model.add(BatchNormalization())#added 20210303

    #added layer with 64 filters 20210303
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))#input_shape, padding removed
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))#input_shape, padding removed
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#strides added 20210303
    model.add(BatchNormalization())#added 20210303

    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#strides added 20210303
    model.add(BatchNormalization())#added 20210303

    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#strides added 20210303
    model.add(BatchNormalization())#added 20210226

#    #added layer with 512 filters 20210303
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#strides added 20210303
    model.add(BatchNormalization())#added 20210226

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))#20210305 0.3 --> 0.5
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation="relu"))
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

    pipeline_from_command_line(create_2D_cnn_model, rgb=False)
