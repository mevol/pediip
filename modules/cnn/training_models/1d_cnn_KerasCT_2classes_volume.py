"""Testing with training_pipeline_3d"""

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from modules.cnn.training_models.training_pipeline_1D_classification import pipeline_from_command_line

def create_1D_cnn_model(input_shape: int):#Tuple[int, int]
  model = Sequential()

  #reduced number of filters 32 --> 16 20210303
  inputs = keras.Input(input_shape)
  x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(inputs)
  x = layers.MaxPool1D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool1D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv1D(filters=128, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool1D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.Conv1D(filters=256, kernel_size=3, activation="relu")(x)
  x = layers.MaxPool1D(pool_size=2)(x)
  x = layers.BatchNormalization()(x)

  x = layers.GlobalAveragePooling1D()(x)
  x = layers.Dense(units=512, activation="relu")(x)
  x = layers.Dropout(0.3)(x)

  outputs = layers.Dense(units=1, activation="sigmoid")(x)

  # Define the model.
  model = keras.Model(inputs, outputs, name="1dcnn")

  print(model.output_shape)

  initial_learning_rate = 0.0001
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
  model.compile(
        loss="binary_crossentropy",#categorical_crossentropy
        #optimizer='adam',
#        optimizer=Adam(learning_rate=0.0001), # was added 20220506
        optimizer=Adam(learning_rate=lr_schedule), # was added 20220513
#        optimizer=SGD(learning_rate=0.0001), # was added 20220509
#        optimizer=SGD(learning_rate=lr_schedule), # was added 20220517
        metrics=["accuracy"],)

  return model


if __name__ == "__main__":

  pipeline_from_command_line(create_1D_cnn_model)#, rgb=False
