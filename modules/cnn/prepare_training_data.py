import argparse
import logging
import os
import yaml
import gemmi

from pathlib import Path
from typing import List

from modules.cnn.delete_temp_files import delete_temp_files

def prepare_training_data(
    mtz_directory: str,
    mtz_file: str,
    xyz_limits: List[int],
    output_directory: str,
    delete_temp: bool = True,
):
    """Convert both the original and inverse hands of a structure into a regular map file based on information
    about the cell info and space group and the xyz dimensions. Return True if no exceptions"""

    logging.info("Preparing training data")

    # Check all directories exist
    try:
        mtz_dir = Path(mtz_directory)
        assert mtz_dir.exists()
    except Exception:
        logging.error(f"Could not find mtz directory at {mtz_directory}")
        raise

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

    # Get lists of child directories
    mtz_structs = [struct.stem for struct in mtz_dir.iterdir()]
    mtz_structs = sorted(mtz_structs)
    logging.debug(f"Following structures found to transform: {mtz_structs}")

    # Get cell info and space group
    cell_info_dict = {}
    space_group_dict = {}

    for struct in mtz_structs:
      struct_dir = Path(os.path.join(mtz_dir, struct))
      homo_lst = [homo.stem for homo in struct_dir.iterdir()] 
      for homo in homo_lst:
        homo_dir = os.path.join(struct_dir, homo)
        logging.info(
          f"Collecting info from {struct}, {mtz_structs.index(struct)+1}/{len(mtz_structs)}")
        if mtz_file in os.listdir(homo_dir):
          print("found MTZ file") 
          logging.info(
            f"Collecting info from {homo}, {homo_lst.index(homo)+1}/{len(homo_lst)}")

          homo_mtz = Path(os.path.join(homo_dir, mtz_file))
          try:
            homo_mtz = Path(os.path.join(homo_dir, mtz_file))
            assert homo_mtz.exists()
          except Exception:
            logging.error(f"Could not find cell info file at {homo_mtz}")
            raise

          try:
            data = gemmi.read_mtz_file(str(homo_mtz))
            cell = data.cell
            print("The MTZ unit cell is: ", cell)
            sg = data.spacegroup
            print("The MTZ space group is: ", sg)
          except Exception:
            logging.error(f"Could not read {homo_mtz}")
            raise
          temp_out_file = os.path.join(output_dir, "temp_"+struct+"_"+homo+".ccp4")
          print(temp_out_file)


          try:
            data_to_map = gemmi.Ccp4Map()
            data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=4)
            data_to_map.update_ccp4_header(2, True)
            data_to_map.write_ccp4_map(temp_out_file) 
          except Exception:
            logging.error(f"Could not create map from {homo_mtz}")
            raise
          
          try: 
            map_to_map = gemmi.read_ccp4_map(temp_out_file)
            print(1111111111, map_to_map.grid)
            map_to_map.grid.unit_cell
            sg = map_to_map.grid.spacegroup
            map_to_map.setup()
           
 
            map_to_grid = map_to_map.grid
            map_to_grid_newcell = map_to_grid.set_unit_cell(gemmi.UnitCell(200, 200, 200, 90, 90, 90))
            #print(map_to_grid_newcell.grid.unit_cell)
            print(333333333, map_to_grid.unit_cell)

            



            #print(map_to_grid_newcell.grid)
            #map_to_grid_newcell.setup()
           # print(map_to_grid_newcell)
            map_to_grid.spacegroup = map_to_map.grid.spacegroup
           # print(map_to_grid.spacegroup)
           # print("The grid of current map is: ", map_to_map.grid)



           # print("The grid unit cell is: ", map_to_map.grid.unit_cell)
           # print("The grid space group is: ", map_to_map.grid.spacegroup)
           # map_to_map = map_to_map.set_unit_cell(gemmi.UnitCell(200, 200, 200, 90, 90, 90))
           # print("The expanded grid unit cell is: ", map_to_map.grid.unit_cell)
           # map_to_map = map_to_map.setup()
           # print("The the updated grid of map is: ", map_to_map.grid)
            #grid = gemmi.FloatGrid(200, 200, 200)
            #map_to_map = map_to_map.grid
            #map_to_map.grid
           # print("The grid of new map is:", map_to_map.grid)
           #ccp4.grid = xmap
          except Exception:
            logging.error(f"Could not expand map {map_to_map}")          
            raise
          #set new extent for 200A, i.e. replace the 5
          #  m.set_extent(st.calculate_fractional_box(margin=5))
          #except Exception:
          #  logging.error(f"Could not expand map for {temp_out_file}")
          #  raise

          #final = os.path.join(output_dir, struct+"_"+homo+".ccp4")
          #try:
          #  m.write_ccp4_map(final)
          #except Exception:
          #  logging.error(f"Could not write final map {final}")
          #write out the standardised map
            1/0


# try:
            

         # try:
         #   cell_info_dict[homo] = mtz_get_cell(homo_mtz)
         # except Exception:
         #   logging.error(f"Could not get cell info from {homo_mtz}")
         #   raise
#          try:
#            space_group_dict[homo] = find_space_group(homo_mtz)
#          except Exception:
#            logging.error(f"Could not get space group from {homo_mtz}")
#            raise

   # logging.info("Collected cell info and space group")

    1/0

    # Set up function to get space group depending on suffix
    if Path(mtz_directory).suffix == ".mtz":
        find_space_group = mtz_find_space_group
    else:
        find_space_group = textfile_find_space_group

    for struct in mtz_structs:
        logging.info(
            f"Collecting info from {struct}, {mtz_structs.index(struct)+1}/{len(mtz_structs)}"
        )

    # Begin transformation
    for struct in mtz_structs:
        logging.info(
            f"Converting {struct}, {mtz_structs.index(struct)+1}/{len(mtz_structs)}"
        )
    if delete_temp is True:
        delete_temp_files(output_directory)
        logging.info("Deleted temporary files in output directory")

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


    if "delete_temp" not in params.keys():
        params["delete_temp"] = True

    return params


def params_from_cmd(args):
    """Extract the parameters for prepare_training_data from the command line and return a dict"""
    params = {
        "mtz_dir": args.mtz_dir,
        "mtz_file": args.mtz_file,
        "xyz_limits": args.xyz,
        "output_dir": args.maps_dir,
        "delete_temp": True,
    }
    if args.keep_temp:
        params["delete_temp"] = False

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
        "mtz_dir", type=str, help="top level directory for mtz information"
    )
    cmd_parser.add_argument(
        "mtz_file", type=str, help="mtz file from specific structure solution step"
    )
    cmd_parser.add_argument(
        "xyz", type=int, nargs=3, help="xyz size of the output map file"
    )
    cmd_parser.add_argument(
        "output_dir", type=str, help="directory to output all map files to"
    )
    cmd_parser.add_argument(
        "--keep_temp",
        action="store_false",
        help="keep the temporary files after processing",
    )
    cmd_parser.set_defaults(func=params_from_cmd)

    # Extract the parameters based on the yaml/command line argument
    args = parser.parse_args()
    parameters = args.func(args)

    print(parameters)

    # Execute the command
    try:
        prepare_training_data(
            parameters["mtz_dir"],
            parameters["mtz_file"],
            parameters["xyz_limits"],
            parameters["maps_dir"],
            parameters["delete_temp"],
        )
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data")
