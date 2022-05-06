"""Testing with training_pipeline_3d"""

from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization #added 20210226
from modules.cnn.training_models.training_pipeline_volume_classification import pipeline_from_command_line

def create_3D_cnn_model(input_shape: Tuple[int, int, int, int]):
  model = Sequential()

  #reduced number of filters 32 --> 16 20210303
  model.add(
      Conv3D(16, kernel_size=(5, 5, 5), strides=(1, 1, 1), padding='same',
             activation="relu", input_shape=input_shape))#kernel 3 --> 5 20210226
  model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

  #added layer with 32 filters 20210303
  model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))#input_shape, padding removed
  #added layer with 32 filters 20210303
  model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))#input_shape, padding removed
  model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))#strides added 20210303
  model.add(BatchNormalization())#added 20210303

  #added layer with 64 filters 20210303
  model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))#input_shape, padding removed
  model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))#input_shape, padding removed
  model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))#strides added 20210303
  model.add(BatchNormalization())#added 20210303

  model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(Conv3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))#strides added 20210303
  model.add(BatchNormalization())#added 20210303

  model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(Conv3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))#strides added 20210303
  model.add(BatchNormalization())#added 20210226

  #added layer with 512 filters 20210303
  model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(Conv3D(512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu", padding='same'))
  model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1)))#strides added 20210303
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
        #optimizer='adam',
        optimizer=Adam(learning_rate=0.0001), # was added 20220506
        metrics=["accuracy"],)

  return model


if __name__ == "__main__":

  pipeline_from_command_line(create_3D_cnn_model, rgb=False)
