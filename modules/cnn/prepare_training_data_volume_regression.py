import argparse
import logging
import os
import yaml
import gemmi
import random

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List

from scipy.ndimage import rotate
from dltk.io.preprocessing import normalise_zero_one
from dltk.io.augmentation import add_gaussian_offset


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

  # opening either a submitted list of files to iterate over or an individual sample
  try:
    data = pd.read_csv(maps_list)
    total_num_maps = len(data)
    logging.info(f"Found {total_num_maps} samples for training")
  except Exception:
    logging.error(f"No LIST of samples provided; working on single sample instead")
    pass
  try:
    data = maps_list
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

    # opening the input MTZ file and convert to map
    try:
      # try opening MTZ file
      data = gemmi.read_mtz_file(str(map_file_path))
      # get reciprocal lattice grid size
      recip_grid = data.get_size_for_hkl()
      logging.info(f"Original size of reciprocal lattice grid: {recip_grid} \n")
      # create an empty map grid
      data_to_map = gemmi.Ccp4Map()
      # turn MTZ file into map using a sample_rate=6; minimal grid size is
      # placed in relation to the resolution, dmin/sample_rate; sample_rate=4
      # keeps the original grid size --> voxel size = 1
      data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=1)
      # set space group to P1
      data_to_map.grid.spacegroup = gemmi.find_spacegroup_by_name('P1')
      # update the map header
      data_to_map.update_ccp4_header(2, True)
      # create a new grid for the updated map
      map_grid = data_to_map.grid
      # create an empty array to write the interpolated density into
      map_array = np.zeros([int(xyz_limits[0]), int(xyz_limits[2]),
                            int(xyz_limits[2])], dtype=np.float32)
      logging.info(f"Grid after expansion of MAP for sample_rate = 1 to {map_array.shape}) ")
      tr = gemmi.Transform()
      tr.mat.fromlist([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      tr.vec.fromlist([int(xyz_limits[0]), int(xyz_limits[2]), int(xyz_limits[2])])
      # finally we calculate interpolated values
      map_grid.interpolate_values(map_array, tr)
      # normalize array
      map_array_normed = normalise_zero_one(map_array)
    except Exception:
      logging.error(f"Could not open MTZ and convert to MAP {map_file_path} \n")
      raise
    try:
      length = int(xyz_limits[0])+1

      if augmentation == True:
        pick_list = [0, 1, 2, 3, 4, 5, 6]
        pick = random.choice(pick_list)

        if pick == 2 or pick == 4:

          deg = np.random.choice(90, 1, replace=False)[0]
          map_array_normed = rotate(map_array_normed, angle = deg, axes=(1, 0), reshape=False)

        if pick == 0 or pick == 6:

          deg = np.random.choice(90, 1, replace=False)[0]
          map_array_normed = rotate(map_array_normed, angle = deg, axes=(1, 2), reshape=False)

        if pick == 3 or pick == 5:
          map_array_normed = add_gaussian_offset(map_array_normed, sigma=0.5)

      logging.info(f"Size of standardise map when finished: {map_array_normed.shape} \n")
    except Exception:
      logging.error(f"Could not expand map {map_file_path} \n")
      raise
  except Exception:
    logging.error(f"Could not open input map list \n")
    raise
  return map_array_normed


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


