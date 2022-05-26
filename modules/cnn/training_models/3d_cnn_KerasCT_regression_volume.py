"""Testing with training_pipeline_3d"""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization #added 20210226
from modules.cnn.training_models.training_pipeline_volume_regression import pipeline_from_command_line

def create_3D_cnn_model(input_shape: Tuple[int, int, int, int]):
  model = Sequential()

  #reduced number of filters 32 --> 16 20210303
  inputs = keras.Input(input_shape)
  x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
  x = layers.MaxPool3D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool3D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool3D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool3D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.GlobalAveragePooling3D()(x)
  x = layers.Dense(units=512, activation="relu")(x)
  x = layers.Dropout(0.3)(x)

  outputs = layers.Dense(units=1, activation="sigmoid")(x)

  # Define the model.
  model = keras.Model(inputs, outputs, name="3dcnn")

  print(model.output_shape)

  initial_learning_rate = 0.0001
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
  model.compile(
        loss='mean_absolute_error',#for regression
        #optimizer='adam',
#        optimizer=Adam(learning_rate=0.0001), # was added 20220506
#        optimizer=Adam(learning_rate=lr_schedule), # was added 20220513
#        optimizer=SGD(learning_rate=0.0001), # was added 20220509
        optimizer=SGD(learning_rate=lr_schedule), # was added 20220517
        metrics = [RootMeanSquaredError()])
#        metrics=["accuracy"],)

  return model


if __name__ == "__main__":

  pipeline_from_command_line(create_3D_cnn_model, rgb=False)
