import argparse
import logging
import os
import yaml
import gemmi
import csv
import re

import numpy as np

from pathlib import Path
from typing import List
from PIL import Image


def slice_map(volume, slices_per_axis):
  """Slice the volume into 2d panes along x, y, z axes and return as an image stack"""
  # Check volume is equal in all directions
  assert (
      volume.shape[0] == volume.shape[1] == volume.shape[2]
  ), f"Please provide a volume which has dimensions of equal length, not {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}"

  length = volume.shape[0]

  # Array to return the images
  image_stack = np.zeros((slices_per_axis * 3, length, length))
  # print(image_stack.shape)

  # Get x slices and put in image_stack
  for slice in range(slices_per_axis):
    image_stack[slice, :, :] = volume[
        (slice + 1) * int((length) / (slices_per_axis + 1)), :, :
    ]

  # Get y slices and put in image_stack
  for slice in range(slices_per_axis):
    image_stack[slice + slices_per_axis, :, :] = volume[
        :, (slice + 1) * int((length) / (slices_per_axis + 1)), :
    ]

    # Get z slices and put in image_stack
  for slice in range(slices_per_axis):
    image_stack[slice + (slices_per_axis * 2), :, :] = volume[
        :, :, (slice + 1) * int((length) / (slices_per_axis + 1))
    ]

  return image_stack

def prepare_training_data_binary(
  maps_list: str,
  xyz_limits: List[int],
  output_directory: str,
  slices_per_axis: int,):
  """Load electron density maps from phasing and slice into 2D images along all three
  axis. Return True if no exceptions"""

  print("Number of slices ", slices_per_axis)

  logging.info("Preparing training data")

  # Check all directories exist
  try:
    output_dir = Path(output_directory)
    assert output_dir.exists()
  except Exception:
    logging.error(f"Could not find output directory at {output_directory}")
    raise

  # Check xyz limits are of correct format
  try:
    assert type(xyz_limits) == list or type(xyz_limits) == tuple
    assert len(xyz_limits) == 3
    assert all(type(values) == int for values in xyz_limits)
  except AssertionError:
    logging.error("xyz_limits muste be provided as a list or tupls of three integer values")
    raise

# #this below works but runs serial
#     with open(maps_list, "r") as ls:
#       next(ls)
#       for line in ls:
#         input_map_path = line.split(",")[0]
#         print(input_map_path)
#         split_path = line.split("/")
#         dir_stem = re.findall(r'\b[a-z0-9]{8}\b-\b[a-z0-9]{4}\b-\b[a-z0-9]{4}\b', line)
# 
#         #Check path to map exists
#         try:
#           map_file_path = Path(input_map_path)
#           print(map_file_path)
#           assert map_file_path.exists()
#         except Exception:
#           logging.error(f"Could not find mtz directory at {map_file_path}")
#           raise
# 
#         try: 
#           opening temporary map file which shouldn't be neccessary to be written out
#           map = gemmi.read_ccp4_map(str(map_file_path))
#         except Exception:
#           logging.error(f"Could not open map {map_file_path}")          
#           raise
#         
#         try:
#           map.setup() 
#           print("Grid after loading temp file", map.grid)
#         except RuntimeError:
#           pass  

#########################################################################################
# a fix to be able to work with the MR data rather than the beamline EP data which is the
# snippet above
  # opening sample list to iterate over
  with open(maps_list, "r") as map_files:
    data_reader = csv.reader(map_files, delimiter=',')
    next(data_reader)
    for sample in data_reader:
      mtz_path = sample[3]#currently needs changing based on column order; pandas way?
      print(mtz_path)
      try:
        os.path.exists(mtz_path)
      except Exception:
        logging.error(f"Could not find MTZ file {mtz_path} \n")
        pass
      try:
        target_file = mtz_path.split("/")[-1]
        print(target_file)
        target_file_stripped = target_file.split(".")[0]
        print(target_file_stripped)
      except Exception:
        logging.error(f"MTZ file stem {target_file_stripped} \n")
        pass
      try:
        target_name = mtz_path.split("/")[8]
        logging.info(f"Working on target: {target_name} \n")
      except Exception:
        pass
      try:
        homo = mtz_path.split("/")[12]
        logging.info(f"Working on homologue: {homo} \n")
      except Exception:
        homo = "none"
        logging.error(f"Could not find homologue to work with. \n")
        pass
      # reading MTZ file with gemmi
      try:
        data = gemmi.read_mtz_file(mtz_path)
      except Exception:
        logging.error(f"Could not read {mtz_path} \n")
        pass
      try:
        # get reciprocal lattice grid size
        recip_grid = data.get_size_for_hkl()
        logging.info(f"Original size of reciprocal lattice grid: {recip_grid} \n")
        # get grid size in relation to resolution and a sample rate of 4
        size1 = data.get_size_for_hkl(sample_rate=6)
        logging.info(f"Reciprocal lattice grid size at sample_rate=6: {size1} \n")
        # create an empty map grid
        map = gemmi.Ccp4Map()
        # turn MTZ file into map using a sample_rate=4; minimal grid size is
        # placed in relation to the resolution, dmin/sample_rate; sample_rate=4
        # doubles the original grid size
        map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=6)#was 4
        map.update_ccp4_header(2, True)
      except Exception:
        logging.error(f"Could not create map from {mtz_path} \n")
        raise
        
##########################################################################################
        
        
      try:
        #this bit here expands the unit cell to be 200A^3;
        #Can I expand the unit cell to standard volume and then extract a
        #grid cube (200, 200, 200) or whatever value has been passed through YAML file
        upper_limit = gemmi.Position(*xyz_limits)
        box = gemmi.FractionalBox()
        box.minimum = gemmi.Fractional(0, 0, 0)
        box.maximum = map.grid.unit_cell.fractionalize(upper_limit)
        box.maximum = map.grid.point_to_fractional(
                    map.grid.get_point(int(xyz_limits[0]),
                                       int(xyz_limits[1]),
                                       int(xyz_limits[2])))
        box.add_margin(1e-5)
        map.set_extent(box)
        map_grid = map.grid
        map_array = np.array(map_grid, copy = False)
        print(map_array.shape)
        print("Grid after setting XYZ limits for MAP", map_grid)
      except Exception:
        logging.error(f"Could not expand map {map_file_path}")
        raise

      try:
        # Slice the volume into images
        image_slices = slice_map(map_array, slices_per_axis)
        # Iterate through images, scale them and save them in output_directory
        for slice_num in range(image_slices.shape[0]):
          # Get slice
          slice = image_slices[slice_num, :, :]
          # Scale slice
          slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
          # Round to the nearest integer
          slice_scaled_int = np.rint(slice_scaled)

          # Save image
          try:
            output_file = Path(output_directory) / Path(f"{dir_stem[0]}_{slice_num}.png")
            Image.fromarray(slice_scaled_int).convert("L").save(output_file)
          except Exception:
            logging.error(f"Could not create image file in {output_directory}")

      except Exception:
        logging.info(f"Finished creating images in {output_directory}")
        raise


#         Check volume is equal in all directions
#         
#         assert (
#           map_array.shape[0] == map_array.shape[1] == map_array.shape[2]
#         ), f"Please provide a volume which has dimensions of equal length, not {map_array.shape[0]}x{volume.shape[1]}x{map_array.shape[2]}"
# 
#         length = map_array.shape[0]
# 
#         Array to return the images
#         image_stack = np.zeros((slices_per_axis * 3, length, length))
#         print(image_stack.shape)
#           
#     Get x slices and put in image_stack
#     for slice in range(slices_per_axis):
#         image_stack[slice, :, :] = volume[
#             (slice + 1) * int((length) / (slices_per_axis + 1)), :, :
#         ]
# 
#     Get y slices and put in image_stack
#     for slice in range(slices_per_axis):
#         image_stack[slice + slices_per_axis, :, :] = volume[
#             :, (slice + 1) * int((length) / (slices_per_axis + 1)), :
#         ]
# 
#     Get z slices and put in image_stack
#     for slice in range(slices_per_axis):
#         image_stack[slice + (slices_per_axis * 2), :, :] = volume[
#             :, :, (slice + 1) * int((length) / (slices_per_axis + 1))
#         ]
# 
#     return image_stack          
          
#          
# #Trying to account for resolution and make the distance between the grid points equal for
# #all resolutions; this causes errors with some space groups
# #          try:
# #            map_to_map = gemmi.read_ccp4_map(temp_out_file)
# #            map_to_map.setup()
# #            print(map_to_map.grid)
# #            grid = map_to_map.grid
# #            print(grid)
# #            new_grid = grid.set_value(200, 200, 200, 4.0)
# #            print(new_grid.get_value)
# #            xyz_limits = [200, 200, 200]
# #            upper_limit = gemmi.Position(*xyz_limits)
# #            box = gemmi.FractionalBox()
# #            box.minimum = gemmi.Fractional(0, 0, 0)
# #            box.maximum = map_to_map.grid.unit_cell.fractionalize(upper_limit)
# #            map_to_map.set_extent(box)
# #          except Exception:
# #            logging.error(f"Could not expand map {map_to_map}")
# #            raise
# 
# 
# 
#           mtz_state = str(mtz_file).strip(".mtz")
#           final_name = struct+"_"+homo+"_"+mtz_state+".ccp4"
#           final = os.path.join(output_dir, final_name)
# #          final = os.path.join(output_dir, struct+"_"+homo+"_"+mtz_state+".ccp4")
#           try:
#             map_to_map.write_ccp4_map(final)
# #            data_to_map.write_ccp4_map(final)
#           except Exception:
#             logging.error(f"Could not write final map {final}")
#               
  return True


def params_from_yaml(args):
  """Extract the parameters for prepare_training_data from a yaml file and return a dict"""
  # Check the path exists
  try:
    config_file_path = Path(args.config_file)
    assert config_file_path.exists()
  except Exception:
    logging.error(f"Could not find config file at {args.config_file}")
    raise

  # Load the data from the config file
  try:
    with open(config_file_path, "r") as f:
      params = yaml.safe_load(f)
  except Exception:
    logging.error(f"Could not extract parameters from yaml file at {config_file_path}")
    raise

  return params


def params_from_cmd(args):
  """Extract the parameters for prepare_training_data from the command line and return a dict"""
  params = {
      "maps_list": args.maps_list,
      "xyz_limits": args.xyz_limits,
      "output_dir": args.output_dir,
      "slices": args.slices_per_axis,}

  return params


if __name__ == "__main__":

  logging.basicConfig(level=logging.INFO)
  log = logging.getLogger(name="debug_log")
  userlog = logging.getLogger(name="usermessages")

  # Parser for command line interface
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers()

  yaml_parser = subparsers.add_parser("yaml")
  yaml_parser.add_argument(
       "config_file",
      type=str,
      help="yaml file with configuration information for this program",)
  yaml_parser.set_defaults(func=params_from_yaml)

  cmd_parser = subparsers.add_parser("cmd")
  cmd_parser.add_argument(
      "maps_list", type=str, help="list of map files to be converted")
  cmd_parser.add_argument(
      "xyz_limits", type=int, nargs=3, help="xyz size of the output map file")
  cmd_parser.add_argument(
      "output_dir", type=str, help="directory to output all map files to")
  cmd_parser.add_argument(
      "slices", type=int, help="number of image slices to produce per axis, default=20",
       default=20)
  cmd_parser.set_defaults(func=params_from_cmd)

  # Extract the parameters based on the yaml/command line argument
  args = parser.parse_args()
  parameters = args.func(args)

  print(parameters)

  # Execute the command
  try:
      prepare_training_data_binary(
          parameters["maps_list"],
          parameters["xyz_limits"],
          parameters["output_dir"],
          parameters["slices"]
      )
  except KeyError as e:
    logging.error(f"Could not find parameter {e} to prepare training data")
