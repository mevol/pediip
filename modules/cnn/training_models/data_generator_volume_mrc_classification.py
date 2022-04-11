import numpy as np
import mrcfile

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

#class DataGenerator(keras.utils.Sequence):
class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32),
               n_classes=2, n_channels=1, shuffle=True):
    'Initialization'
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs_new = np.arange(len(list_IDs))
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
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
    X = np.empty((self.batch_size,
                  *self.dim,
                  self.n_channels))

    #this version was used in python3-topaz
    y = np.empty([self.batch_size], dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      with mrcfile.open(self.list_IDs.iloc[ID, 0]) as mrc:
        volume = mrc.data

      # Store sample
      X[i,] = volume.reshape(*self.dim, self.n_channels)

      y[i] = self.labels[ID]

    X = X.reshape(self.batch_size, *self.dim, self.n_channels)
    return X, to_categorical(y, num_classes=self.n_classes) 

