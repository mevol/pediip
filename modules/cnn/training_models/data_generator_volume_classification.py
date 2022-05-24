import numpy as np
import gemmi

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from modules.cnn.prepare_training_data_volume import prepare_training_data_volume


class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, xyz_limits, list_IDs, labels, batch_size=32, dim=(32,32,32),
               n_classes=2, n_channels=1, shuffle=True, augmentation=False):
    'Initialization'
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs_new = np.arange(len(list_IDs))
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.augmentation = augmentation
    self.xyz_limits = xyz_limits
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs_new) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # Find list of IDs
    list_IDs_temp = [self.list_IDs_new[k] for k in index]
    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs_new))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.zeros((self.batch_size,
                  *self.dim,
                  self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
#      sample = self.list_IDs.iloc[ID, :]
      sample = self.list_IDs.iloc[ID, 0]
      path = sample["filename"]
      #protocol = sample["protocol"]
      #stage = sample["stage"]
      # Store sample
      volume = prepare_training_data_volume(path,
                                            self.xyz_limits,
                                            self.augmentation)
      X[i,] = volume.reshape(*self.dim, self.n_channels)
      # Store class
      y[i] = self.labels[ID]
      
#    print(X)
#    print(y)
    #return X, to_categorical(y, num_classes=self.n_classes)
    return X, y

