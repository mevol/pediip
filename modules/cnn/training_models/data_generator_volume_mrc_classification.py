import numpy as np
import mrcfile
import os

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from scipy.ndimage import rotate

class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, labels, map_dir, batch_size=32, dim=(32,32,32),
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
    self.map_dir = map_dir
    self.on_epoch_end()
#    self.replace_filename()

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

  def __replace_filename(self, x):
    try:
      target_file = x.split("/")[-1]
      target_file_stripped = target_file.split(".")[0]
    except Exception:
      pass
    try:
      target_name = x.split("/")[8]
    except Exception:
      pass
    try:
      homo = x.split("/")[12]
    except Exception:
      homo = "none"
      pass
    sample_path = os.path.join(self.map_dir,
                          target_name+"_"+homo+"_"+target_file_stripped+".ccp4")
    try:
      # expand this path to its real path as it is a sym link pointing to my local,
      # hand-crafted PDB-redo version; this one has the same subfolder arrangement
      # as my local PDB version; makes traversing easier; however, in order to create
      # this custom PDB-redo version I created again sym links to the original
      # PDB-redo; hence I need two levels to expand the real file path
      real_input_path = os.path.realpath(sample_path)
      # replace "/dls/" with "/opt/" to read files in the mount pount
      real_input_path_opt = real_input_path.replace("/dls/", "/opt/")
      # expand the next level of sym link to the real path
      real_path_to_map = os.path.realpath(real_input_path_opt)
      # replace "/dls/" with "/opt/" to read files in the mount pount
      real_path_to_map_opt = real_path_to_map.replace("/dls/", "/opt/")
      map_file_path = Path(os.path.realpath(real_path_to_map_opt))
    except Exception:
      pass
    return map_file_path

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
      # add a function to convert file path self.list_IDs.iloc[ID, 0]
      # so it can be found on disk from within the container;
      # the function is currently in training pipeline and iterates over
      # a pandas dataframe --> bad
      with mrcfile.open(self.__replace_filename(self.list_IDs.iloc[ID, 0]), mode='r+') as mrc:
        mrc.voxel_size = 1.0
        volume = mrc.data

      # normalise
      array_max = np.max(volume)
      array_min = np.min(volume)
      diff = array_max - array_min
      volume_normed = ((volume - array_min) / diff)#map_array

      volume_normed[volume_normed < 0] = 0
      volume_normed[volume_normed > 1] = 1

      length = volume_normed.shape[0]
      edited_volume = np.zeros((length, length, length))
      # rotate entire volume
      deg = np.random.choice(90, 1, replace=False)[0]
      volume_rot = rotate(volume_normed, deg, reshape=False)

#      # rotation around x-axis
#      deg1 = np.random.choice(90, 1, replace=False)[0]
#      for slice_num in range(volume.shape[0]):
#        #print("Working on slice number: ", slice_num)
#        # Get slice
#        slice = volume[slice_num, :, :]
#        # Scale slice
#        slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
#        # Round to the nearest integer
#        slice_scaled_int = np.rint(slice_scaled)
#        # do data augmentation as rotation for a random angle between 0 and 90 deg
#        # for all even numbers in the total image stack
#        # check that the remainder of division is 0 and hence the result even
#        # get a random number between 0 and 90 deg
#        # rotate the slice by this deg
#        slice_scaled_int = rotate(slice_scaled_int, angle = deg1, reshape=False)
#        # combine the slices to a new image stack for training
#        edited_volume[slice_num, :, :] = slice_scaled_int
#
#      # rotation around y-axis
#      deg2 = np.random.choice(90, 1, replace=False)[0]
#      for slice_num in range(volume.shape[1]):
#        #print("Working on slice number: ", slice_num)
#        # Get slice
#        slice = volume[:, slice_num, :]
#        # Scale slice
#        slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
#        # Round to the nearest integer
#        slice_scaled_int = np.rint(slice_scaled)
#        # do data augmentation as rotation for a random angle between 0 and 90 deg
#        # for all even numbers in the total image stack
#        # check that the remainder of division is 0 and hence the result even
#        # get a random number between 0 and 90 deg
#        # rotate the slice by this deg
#        slice_scaled_int = rotate(slice_scaled_int, angle = deg2, reshape=False)
#        # combine the slices to a new image stack for training
#        edited_volume[:, slice_num, :] = slice_scaled_int
#
#      # rotation around z-axis
#      deg3 = np.random.choice(90, 1, replace=False)[0]
#      for slice_num in range(volume.shape[1]):
#        #print("Working on slice number: ", slice_num)
#        # Get slice
#        slice = volume[:, :, slice_num]
#        # Scale slice
#        slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
#        # Round to the nearest integer
#        slice_scaled_int = np.rint(slice_scaled)
#        # do data augmentation as rotation for a random angle between 0 and 90 deg
#        # for all even numbers in the total image stack
#        # check that the remainder of division is 0 and hence the result even
#        # get a random number between 0 and 90 deg
#        # rotate the slice by this deg
#        slice_scaled_int = rotate(slice_scaled_int, angle = deg3, reshape=False)
#        # combine the slices to a new image stack for training
#        edited_volume[:, :, slice_num] = slice_scaled_int

      # Store sample
#      X[i,] = edited_volume.reshape(*self.dim, self.n_channels)
      X[i,] = volume_rot.reshape(*self.dim, self.n_channels)

      y[i] = self.labels[ID]

    X = X.reshape(self.batch_size, *self.dim, self.n_channels)
    return X, to_categorical(y, num_classes=self.n_classes) 

