import numpy as np
#import mrcfile
import gemmi

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from modules.cnn.prepare_training_data_random_pick_combined import prepare_training_data_random_pick_combined


class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32),
               n_classes=2, n_channels=1, shuffle=True):
    'Initialization'
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.iterator = self.list_IDs.index.tolist()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    #print("All indexes ", index)
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    #print("Indexes of batch ", indexes)
    # Find list of IDs
    list_IDs_temp = [self.iterator[k] for k in indexes]
    print("List of IDs ", self.iterator)
    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))

    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #this version was used in python3-topaz
    y = np.empty([self.batch_size], dtype=int)
    
    # Generate data

## TO DO: This should go into the data generator; probably need to do a new one
#    prepare_training_data_random_pick_combined(parameters_dict["sample_lable_lst"],
#                                               parameters_dict["xyz_limits"],
#                                               output_dir_path,
#                                               parameters_dict["slices_per_axis"])





    for i, ID in enumerate(list_IDs_temp):
      print(777, ID)#is the path to the ccp4 map to open
      print(i)#is the index within the chosen batch size
    
      #print("Full list of IDs", self.list_IDs)

      print(99999999999999, X[i,])
 
#      with mrcfile.open(self.list_IDs[ID]) as mrc:
#        volume = mrc.data
#        #TO DO: add standardisation or normalisation here for each volume
#        #print("Loaded dimensions", volume.shape)
#      # Store sample
#      X[i,] = volume.reshape(*self.dim, self.n_channels)

#    for i, ID in enumerate(list_labels_temp):
#      print(888, ID)
#      print(i)
#      print("Stored labels", y)
      # Store class
      #y[i] = ID#this one to use of one-hot encoding within the datagenerator
      #y = ID#this one to use if y had one-hot encoding set in training_pipeline_3d.py
      y[i] = self.labels[ID]

    print("Lable shape", y.shape)  
    X = X.reshape(self.batch_size, *self.dim, self.n_channels)
    X = volume.reshape(self.batch_size, *self.dim, self.n_channels)
    #one-hot encoding on the fly
    return X, keras.utils.to_categorical(y, num_classes=self.n_classes) 
