import argparse
import logging
import os
import yaml
import gemmi
import csv

from pathlib import Path
from typing import List

def prepare_training_data(
  maps_list: str,
  xyz_limits: List[int],
  output_directory: str):
  """Convert both the original and inverse hands of a structure into a regular map file based on information
  about the cell info and space group and the xyz dimensions. Return True if no exceptions"""

  logging.info("Preparing training data \n")

  # Check all directories exist
  try:
    output_dir = Path(output_directory)
    assert output_dir.exists()
  except Exception:
    logging.error(f"Could not find output directory at {output_directory} \n")
    raise

  # Check xyz limits are of correct format
  try:
    assert type(xyz_limits) == list or type(xyz_limits) == tuple
    assert len(xyz_limits) == 3
    assert all(type(values) == int for values in xyz_limits)
  except AssertionError:
    logging.error(
        "xyz_limits muste be provided as a list or tupls of three integer values \n")
    raise

  # check sample list exists
  try:
    assert os.path.exists(maps_list)
  except Exception:
    logging.error(f"No LIST of samples provided; working on single sample instead \n")
    print("Colud not find input LIST")
    pass

  # opening sample list to iterate over
  with open(maps_list, "r") as map_files:
    data_reader = csv.reader(map_files, delimiter=',')
    next(data_reader)
    for sample in data_reader:
      mtz_path = sample[4]# this should be fixed; perhaps turn into a dataframe for better indexing
      print(55555555, mtz_path)
      try:
        os.path.exists(mtz_path)
      except Exception:
        logging.error(f"Could not find MTZ file {mtz_path} \n")
        pass
      try:
        target_file = mtz_path.split("/")[-1]
        target_file_stripped = target_file.split(".")[0]
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
        data_to_map = gemmi.Ccp4Map()
        # turn MTZ file into map using a sample_rate=4; minimal grid size is
        # placed in relation to the resolution, dmin/sample_rate; sample_rate=4
        # doubles the original grid size
        data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=6)#was 4
        data_to_map.update_ccp4_header(2, True)
      except Exception:
        logging.error(f"Could not create map from {mtz_path} \n")
        raise
      try:
        #this bit here expands the unit cell to standard volume and then extract a
        #grid cube of whatever value for xyz_limits has been passed through YAML file
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
        logging.info(f"Reciprocal lattice grid size after standardization : {map_grid} \n")
      except Exception:
        logging.error(f"Could not expand map {data_to_map} \n")
        raise
      # writing out the new, standardized CCP4 map
      try:
        final = os.path.join(output_dir,
                                   target_name+"_"+homo+"_"+target_file_stripped+".ccp4")
        data_to_map.write_ccp4_map(final)
      except Exception:
        logging.error(f"Could not write final map {final} \n")
  return


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
    logging.error(f"Could not extract parameters from yaml file at {config_file_path} \n")
    raise

  return params


def params_from_cmd(args):
  """Extract the parameters for prepare_training_data from the command line and return a dict"""
  params = {
        "maps_list": args.maps_list,
        "xyz_limits": args.xyz_limits,
        "output_dir": args.output_dir,}

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
        "maps_list",
        type=str,
        help="list of MTZ files to convert to CCP4 maps")
  cmd_parser.add_argument(
        "xyz_limits",
        type=int,
        nargs=3,
        help="xyz size of the output map file")
  cmd_parser.add_argument(
        "output_dir",
        type=str,
        help="directory to output all map files to")
  cmd_parser.set_defaults(func=params_from_cmd)

  # Extract the parameters based on the yaml/command line argument
  args = parser.parse_args()
  parameters = args.func(args)

  # Execute the command
  try:
    prepare_training_data(
            parameters["maps_list"],
            parameters["xyz_limits"],
            parameters["output_dir"],)
  except KeyError as e:
    logging.error(f"Could not find parameter {e} to prepare training data")
