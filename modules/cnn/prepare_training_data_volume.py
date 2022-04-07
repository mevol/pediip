import argparse
import logging
import os
import yaml
import gemmi
import csv

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from typing import List
from sys import getsizeof

from scipy.spatial.transform import Rotation as R
from scipy import ndimage

#def rotation(volume):
#  r = R.from_euler('xyz', [
#                           [np.random.choice(90, 1, replace=False), 0, 0],
#                           [0, np.random.choice(90, 1, replace=False), 0],
#                           [0, 0, np.random.choice(90, 1, replace=False)]])
#  print(r)
#  rot_volume = r.apply(volume)
#  return rot_volume

def rotate_man(self, deg_angle, axis):
  d = len(self.matrix)
  h = len(self.matrix[0])
  w = len(self.matrix[0][0])
  min_new_x = 0
  max_new_x = 0
  min_new_y = 0
  max_new_y = 0
  min_new_z = 0
  max_new_z = 0
  new_coords = []
  angle = radians(deg_angle)

  for z in range(d):
    for y in range(h):
      for x in range(w):
        new_x = None
        new_y = None
        new_z = None

        if axis == "x":
          new_x = int(round(x))
          new_y = int(round(y*cos(angle) - z*sin(angle)))
          new_z = int(round(y*sin(angle) + z*cos(angle)))
        elif axis == "y":
          new_x = int(round(z*sin(angle) + x*cos(angle)))
          new_y = int(round(y))
          new_z = int(round(z*cos(angle) - x*sin(angle)))
        elif axis == "z":
          new_x = int(round(x*cos(angle) - y*sin(angle)))
          new_y = int(round(x*sin(angle) + y*cos(angle)))
          new_z = int(round(z))
        val = self.matrix.item((z, y, x))
        new_coords.append((val, new_x, new_y, new_z))
        if new_x < min_new_x: min_new_x = new_x
        if new_x > max_new_x: max_new_x = new_x
        if new_y < min_new_y: min_new_y = new_y
        if new_y > max_new_y: max_new_y = new_y
        if new_z < min_new_z: min_new_z = new_z
        if new_z > max_new_z: max_new_z = new_z

  new_x_offset = abs(min_new_x)
  new_y_offset = abs(min_new_y)
  new_z_offset = abs(min_new_z)

  new_width = abs(min_new_x - max_new_x)
  new_height = abs(min_new_y - max_new_y)
  new_depth = abs(min_new_z - max_new_z)

  rotated = np.empty((new_depth + 1, new_height + 1, new_width + 1))
  rotated.fill(0)
  for coord in new_coords:
    val = coord[0]
    x = coord[1]
    y = coord[2]
    z = coord[3]

    if rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] == 0:
      rotated[new_z_offset + z][new_y_offset + y][new_x_offset + x] = val

  self.matrix = rotated
  return self.matrix

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""
    def scipy_rotate(volume):
        print(1111111, volume.shape)
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        #angle = random.choice(angles)
        angle = np.random.choice(90, 1, replace=False)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        print(2222222, volume.shape)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        print(3333333, volume.shape)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def prepare_training_data_volume(
    maps_list: str,
    xyz_limits: List[int],
    augmentation = False):
    """Load electron density maps from phasing and slice into 2D images along all three
    axis. Return True if no exceptions"""
    logging.info("Preparing training data. \n")

    # Check xyz limits are of correct format
    try:
        assert type(xyz_limits) == list or type(xyz_limits) == tuple
        assert len(xyz_limits) == 3
        assert all(type(values) == int for values in xyz_limits)
    except AssertionError:
        logging.error(
        f"xyz_limits muste be provided as a list or tupls of three integer values \n")
        raise

#this below works but runs serial

    try:
        data = pd.read_csv(maps_list)
        total_num_maps = len(data)
        logging.info(f"Found {total_num_maps} samples for training")
    except Exception:
        logging.error(f"No LIST of samples provided; working on single sample instead")
        pass

    try:
        data = maps_list
        total_bytes = 0
        number_maps = 0

        input_path = data
        # expand this path to its real path as it is a sym link pointing to my local,
        # hand-crafted PDB-redo version; this one has the same subfolder arrangement
        # as my local PDB version; makes traversing easier; however, in order to create
        # this custom PDB-redo version I created again sym links to the original
        # PDB-redo; hence I need two levels to expand the real file path
        real_input_path = os.path.realpath(input_path)
        # replace "/dls/" with "/opt/" to read files in the mount pount
        real_input_path_opt = real_input_path.replace("/dls/", "/opt/")
        # expand the next level of sym link to the real path
        real_path_to_map = os.path.realpath(real_input_path_opt)
        # replace "/dls/" with "/opt/" to read files in the mount pount
        real_path_to_map_opt = real_path_to_map.replace("/dls/", "/opt/")
        try:
            target = input_map_path.split("/")[8]
            logging.info(f"Working on target: {target} \n")
        except Exception:
            pass
        try:
            homo = input_map_path.split("/")[12]
            logging.info(f"Working on homologue: {homo} \n")
        except Exception:
            logging.error(f"Could not find homologue to work with. \n")
            pass
        # Check path to map exists
        try:
            map_file_path = Path(os.path.realpath(real_path_to_map_opt))
            assert map_file_path.exists()
        except Exception:
            logging.error(f"Could not find mtz directory at {map_file_path} \n")
            pass

        try:
            # try opening MTZ file
            data = gemmi.read_mtz_file(str(map_file_path))
            # get reciprocal lattice grid size
            recip_grid = data.get_size_for_hkl()
            logging.info(f"Original size of reciprocal lattice grid: {recip_grid} \n")
            # get grid size in relation to resolution and a sample rate of 4
            size1 = data.get_size_for_hkl(sample_rate=4)
            logging.info(f"Reciprocal lattice grid size at sample_rate=4: {size1} \n")
            # create an empty map grid
            data_to_map = gemmi.Ccp4Map()
            # turn MTZ file into map using a sample_rate=4; minimal grid size is
            # placed in relation to the resolution, dmin/sample_rate; sample_rate=4
            # doubles the original grid size
            data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT',
                                                               sample_rate=4)
            data_to_map.update_ccp4_header(2, True)
        except Exception:
            logging.error(f"Could not open MTZ and convert to MAP {map_file_path} \n")
            raise
        try:
            #this bit here expands the unit cell to be 200A^3;
            #Can I expand the unit cell to standard volume and then extract a
            #grid cube (200, 200, 200) or whatever value has been passed through YAML file
            upper_limit = gemmi.Position(*xyz_limits)
            box = gemmi.FractionalBox()
            box.minimum = gemmi.Fractional(0, 0, 0)
            box.maximum = data_to_map.grid.unit_cell.fractionalize(upper_limit)
            box.maximum = data_to_map.grid.point_to_fractional(
                                          data_to_map.grid.get_point(int(xyz_limits[0]),
                                                                     int(xyz_limits[1]),
                                                                     int(xyz_limits[2])))
            box.add_margin(1e-5)
            data_to_map.set_extent(box)
            map_grid = data_to_map.grid
            map_array = np.array(map_grid, copy = False)

            length = int(xyz_limits[0])+1
            edited_volume = np.zeros((length,
                                            length,
                                            length))
            print("empty edited volume: ", edited_volume.shape)
            # Slice the volume into images
            #image_slices, bytes = slice_map(map_array, slices_per_axis)
            # Iterate through images, scale them and save them in output_directory
            deg = np.random.choice(90, 1, replace=False)[0]
            for slice_num in range(map_array.shape[0]):
                #print("Working on slice number: ", slice_num)
                # Get slice
                slice = map_array[slice_num, :, :]
                # Scale slice
                slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
                # Round to the nearest integer
                slice_scaled_int = np.rint(slice_scaled)
                # do data augmentation as rotation for a random angle between 0 and 90 deg
                # for all even numbers in the total image stack
                # check that the remainder of division is 0 and hence the result even
                # get a random number between 0 and 90 deg
                # rotate the slice by this deg
                slice_scaled_int = rotate(slice_scaled_int, angle = deg, reshape=False)
                # combine the slices to a new image stack for training
                edited_volume[slice_num, :, :] = slice_scaled_int
            print("filled edited volume: ", edited_volume.shape)



#            rotated_volume = rotation(map_array)

#            rotated_volume = rotate_man(map_array, 20, x)
#            rotated_volume = rotate(map_array)

            logging.info(f"Size of standardise map when finished: {map_array.shape} \n")
        except Exception:
            logging.error(f"Could not expand map {map_file_path} \n")
            raise
            total_bytes = total_bytes + bytes
            number_maps = number_maps + 1
            logging.info(f"Accumulated byte size: {total_bytes} \n")
            logging.info(f"Total number of maps processed: {number_maps} \n")

        except Exception:
            raise
    except Exception:
        logging.error(f"Could not open input map list \n")
        raise
#    return map_array
#    return rotated_volume
    return edited_volume


def params_from_yaml(args):
    """Extract the parameters for prepare_training_data from a yaml file and return a dict"""
    # Check the path exists
    try:
        config_file_path = Path(args.config_file)
        assert config_file_path.exists()
    except Exception:
        logging.error(f"Could not find config file at {args.config_file} \n")
        raise

    # Load the data from the config file
    try:
        with open(config_file_path, "r") as f:
            params = yaml.safe_load(f)
    except Exception:
        logging.error(
            f"Could not extract parameters from yaml file at {config_file_path} \n")
        raise

    return params


def params_from_cmd(args):
    """Extract the parameters for prepare_training_data from the command line and return a dict"""
    params = {"maps_list": args.maps_list,
              "xyz_limits": args.xyz_limits}

    return params


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, filename="preparing_data.log", filemode="w")
    log = logging.getLogger(name="debug_log")
    userlog = logging.getLogger(name="usermessages")

    # Parser for command line interface
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        "config_file",
        type=str,
        help="yaml file with configuration information for this program")
    yaml_parser.set_defaults(func=params_from_yaml)

    cmd_parser = subparsers.add_parser("cmd")
    cmd_parser.add_argument(
        "maps_list",
        type=str,
        help="list of map files to be converted")
    cmd_parser.add_argument(
        "xyz_limits",
        type=int,
        nargs=3,
        help="xyz size of the output map file")
    cmd_parser.set_defaults(func=params_from_cmd)

    # Extract the parameters based on the yaml/command line argument
    args = parser.parse_args()
    parameters = args.func(args)

    # Execute the command
    try:
        prepare_training_data_volume(
            parameters["maps_list"],
            parameters["xyz_limits"])
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data \n")
