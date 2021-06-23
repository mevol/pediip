import argparse
import logging
import os
import yaml
import gemmi
import csv

from pathlib import Path
from typing import List

def prepare_training_data_binary(
    maps_list: str,
    xyz_limits: List[int],
    output_directory: str,
):
    """Load electron density maps from phasing and slice into 2D images along all three
    axis. Return True if no exceptions"""

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
        logging.error(
            "xyz_limits muste be provided as a list or tupls of three integer values"
        )
        raise

#this below works but runs serial
    with open(maps_list, "r") as ls:
      next(ls)
      for line in ls:
        print(line)
        line_splitted = line.split(",")[0]
        print(line_splitted)
        
        
        # Check path to map exists
        try:
          map_file_path = Path(line_splitted)
          print(map_file_path)
          assert map_file_path.exists()
        except Exception:
          logging.error(f"Could not find mtz directory at {map_file_path}")
          raise

        try: 
          # opening temporary map file which shouldn't be neccessary to be written out
          map_to_map = gemmi.read_ccp4_map(map_file_path)
          map_to_map.setup()
 
          print("Grid after loading temp file", map_to_map.grid)
# 
#             #this bit here expands the unit cell to be 200A^3;
#             #Can I expand the unit cell to standard volume and then extract a
#             #grid cube (200, 200, 200)
# #            xyz_limits = [200, 200, 200]
# #            xyz_limits = [100, 100, 100]
#             xyz_limits = [50, 50, 50]
#             upper_limit = gemmi.Position(*xyz_limits)
#             box = gemmi.FractionalBox()
#             box.minimum = gemmi.Fractional(0, 0, 0)
#             box.maximum = map_to_map.grid.unit_cell.fractionalize(upper_limit)
# #            box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(200, 200, 200))
# #            box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(100, 100, 100))
#             box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(50, 50, 50))
#             box.add_margin(1e-5)
#             map_to_map.set_extent(box)
# 
#             print("Grid after setting XYZ limits for MAP", map_to_map.grid)
# 
#             #create a grid with extend x=0-->200, y=0-->200, z=0-->200
#             #currently problems as the 200 limit not always reached for all axes;
#             #adding a margin maybe that will help
#            # new_map.setup()
#            # box1 = gemmi.FractionalBox()
#            # box1.minimum = gemmi.Fractional(0, 0, 0)
#            # box1.maximum = new_map.grid.point_to_fractional(new_map.grid.get_point(200, 200, 200))
#            # map_to_map.setup()
#            # new_map.set_extent(box1)
# 
#            # print("Grid after setting grid dimensions", new_map.grid)
# 
        except Exception:
          logging.error(f"Could not expand map {map_to_map}")          
          raise
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
        logging.error(
            f"Could not extract parameters from yaml file at {config_file_path}"
        )
        raise

    return params


def params_from_cmd(args):
    """Extract the parameters for prepare_training_data from the command line and return a dict"""
    params = {
        "maps_list": args.maps_list,
        "xyz_limits": args.xyz_limits,
        "output_dir": args.output_dir,
    }

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
        help="yaml file with configuration information for this program",
    )
    yaml_parser.set_defaults(func=params_from_yaml)

    cmd_parser = subparsers.add_parser("cmd")
    cmd_parser.add_argument(
        "maps_list", type=str, help="list of map files to be converted"
    )
    cmd_parser.add_argument(
        "xyz_limits", type=int, nargs=3, help="xyz size of the output map file"
    )
    cmd_parser.add_argument(
        "output_dir", type=str, help="directory to output all map files to"
    )
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
        )
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data")
