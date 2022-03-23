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
    self.list_IDs = list_IDs
    print(self.list_IDs) #passed correctly
    self.iterator = self.list_IDs.index.tolist()
    print(self.iterator) #passed correctly
    self.n_channels = n_channels
    #print(self.n_channels) #passed correctly
    self.n_classes = n_classes
    #print(self.n_classes) #passed correctly
    self.shuffle = shuffle
    #print(self.shuffle) #passed correctly
    self.xyz_limits = xyz_limits
    print(self.xyz_limits)
    self.slices_per_axis = slices_per_axis
    print(self.slices_per_axis)
    self.on_epoch_end()
    self.n = len(self.list_IDs)

#############################################################
# parameters are passed correctly; why does it not run

  def __len__(self):
    number_of_batches = self.n // self.batch_size
    print("Number of batches to process: ", number_of_batches)
    return number_of_batches

  def on_epoch_end(self):
    if self.shuffle:
      self.list_IDs = self.list_IDs.sample(frac=1).reset_index(drop=True)

#  def __get_output(self, y, num_classes):
#    return keras.utils.to_categorical(y, num_classes=self.n_classes)
#    return tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        
  def __get_data(self, batches):
    X_batch = prepare_training_data_random_pick_combined(batches,
                                               self.xyz_limits,
#                                               output_dir_path,
                                               self.slices_per_axis)
    print(X.shape)
#    name_batch = batches[self.y_col['name']]
#    y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
#    y = self.labels[ID]
    return X_batch, 

  def __getitem__(self, index):
    batches = self.df[list_IDs * self.batch_size:(index + 1) * self.batch_size]
    print(batches)
    X, y = self.__get_data(batches)
    return X, y


#############################################################
#  def __len__(self):
#    'Denotes the number of batches per epoch'
#    print("Length of list to iterate over: ", len(self.list_IDs))
#    print("Length of each batch: ", int(np.floor(len(self.list_IDs) / self.batch_size)))
#    return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#  def __getitem__(self, index):
#    'Generate one batch of data'
#    # Generate indexes of the batch
#    print("All indexes ", index)
#    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#    print("Indexes of batch ", indexes)
#    # Find list of IDs
#    list_IDs_temp = [self.iterator[k] for k in indexes]
#    print("List of IDs ", self.iterator)
#    # Generate data
#    X, y = self.__data_generation(list_IDs_temp)
#    return X, y
#
#  def on_epoch_end(self):
#    'Updates indexes after each epoch'
#    self.indexes = np.arange(len(self.list_IDs))
#    if self.shuffle == True:
#      np.random.shuffle(self.indexes)
#
#  def __data_generation(self, list_IDs_temp):
#    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#    # Initialization
#    X = np.empty((self.batch_size, *self.dim, self.n_channels))
#    #this version was used in python3-topaz
#    y = np.empty([self.batch_size], dtype=int)
#    
    # Generate data






#    for i, ID in enumerate(list_IDs_temp):
#      print(777, ID)#is the path to the ccp4 map to open
#      print(i)#is the index within the chosen batch size
#    
#      #print("Full list of IDs", self.list_IDs)
#
#      print(99999999999999, X[i,])
 
## TO DO: This should go into the data generator; probably need to do a new one
#    prepare_training_data_random_pick_combined(X[i,],
#                                               self.xyz_limits,
#                                               output_dir_path,
#                                               self.slices_per_axis)
#      X[i,] = volume.reshape(*self.dim, self.n_channels)

#    for i, ID in enumerate(list_labels_temp):
#      print(888, ID)
#      print(i)
#      print("Stored labels", y)
      # Store class
      #y[i] = ID#this one to use of one-hot encoding within the datagenerator
      #y = ID#this one to use if y had one-hot encoding set in training_pipeline_3d.py
#      y[i] = self.labels[ID]

#    print("Lable shape", y.shape)
#    X = X.reshape(self.batch_size, *self.dim, self.n_channels)
#    X = volume.reshape(self.batch_size, *self.dim, self.n_channels)
    #one-hot encoding on the fly
#    return X, keras.utils.to_categorical(y, num_classes=self.n_classes) 
