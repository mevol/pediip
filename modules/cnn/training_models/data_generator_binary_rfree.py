import numpy as np
#import mrcfile
import gemmi

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from modules.cnn.prepare_training_data_random_pick_combined import prepare_training_data_random_pick_combined


class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, xyz_limits, slices_per_axis, list_IDs, labels, batch_size=32, dim=(32,32),
               n_classes=2, n_channels=1, shuffle=True):
    'Initialization'
    self.dim = dim
    print(self.dim) #passed correctly
    self.batch_size = batch_size
    #print(self.batch_size) #passed correctly
    self.labels = labels
    #print(self.labels) #passed correctly
    self.list_IDs_new = np.arange(len(list_IDs))
    print("Sample list new: ", self.list_IDs_new) #passed correctly
    self.n_channels = n_channels
    #print(self.n_channels) #passed correctly
    self.n_classes = n_classes
    #print(self.n_classes) #passed correctly
    self.shuffle = shuffle
    #print(self.shuffle) #passed correctly
    self.xyz_limits = xyz_limits
    #print(self.xyz_limits)
    self.slices_per_axis = slices_per_axis
    #print(self.slices_per_axis)
    self.on_epoch_end()

#############################################################
# parameters are passed correctly; why does it not run

  def __len__(self):
    'Denotes the number of batches per epoch'
    print("Number of batches to run: ", int(np.floor(len(self.list_IDs_new) / self.batch_size)))
    return int(np.floor(len(self.list_IDs_new) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    print("Index of batch ", index)
    print("Length of indexes: ", len(self.indexes))

    # Generate indexes of the batch
    index = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    print("range of indexes: ", index)

    # Find list of IDs
    list_IDs_temp = [self.list_IDs_new[k] for k in index]
    print("List of temporary IDs ", list_IDs_temp)

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
    print("Size of empty array: ", X.shape)
    y = np.empty((self.batch_size), dtype=int)

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
#      print("Sample index: ", i)
      print("Sample path: ", ID)
      sample = self.list_IDs.iloc[ID, :]
      print(sample)
      path = sample["filename"]
      protocol = sample["protocol"]
      stage = sample["stage"]
      # Store sample
      stack = prepare_training_data_random_pick_combined(path,
                                               self.xyz_limits,
                                               self.slices_per_axis)
      print("IMAGE STACK SHAPE: ", stack.shape)
      X[i,] = stack.reshape(*self.dim, self.n_channels)


      # Store class
      y[i] = self.labels[ID]

    return X, to_categorical(y, num_classes=self.n_classes)

