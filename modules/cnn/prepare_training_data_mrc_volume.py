import argparse
import logging
import os
import yaml
import gemmi
import csv

from pathlib import Path
from typing import List

from modules.cnn.delete_temp_files import delete_temp_files

def prepare_training_data(
    maps_list: str,
    xyz_limits: List[int],
    output_directory: str,
    delete_temp: bool = True,
):
    """Convert both the original and inverse hands of a structure into a regular map file based on information
    about the cell info and space group and the xyz dimensions. Return True if no exceptions"""

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

    # opening sample list to iterate over
    try:
        print(maps_list)
        assert os.path.exists(maps_list)
    except Exception:
        logging.error(f"No LIST of samples provided; working on single sample instead")
        pass

#this below works but runs serial
    with open(maps_list, "r") as map_files:
        data_reader = csv.reader(map_files, delimiter=',')
        print(data_reader)
        next(data_reader)

        for sample in data_reader:
            print(sample)
            mtz_path = sample[1]
            try:
              os.path.exists(mtz_path)
            except Exception:
              logging.error(f"Could not find MTZ file {mtz_path}")
            pass
            try:
              target = mtz_path.split("/")[8]
              print(target)
              logging.info(f"Working on target: {target} \n")
            except Exception:
              pass
            try:
              homo = mtz_path.split("/")[12]
              print(homo)
              logging.info(f"Working on homologue: {homo} \n")
            except Exception:
              homo = "none"
              logging.error(f"Could not find homologue to work with. \n")
              pass

            try:
              data = gemmi.read_mtz_file(mtz_path)
            except Exception:
              logging.error(f"Could not read {mtz_path}")
            pass
            try:
              # get reciprocal lattice grid size
              recip_grid = data.get_size_for_hkl()
              logging.info(f"Original size of reciprocal lattice grid: {recip_grid} \n")
              # get grid size in relation to resolution and a sample rate of 4
              size1 = data.get_size_for_hkl(sample_rate=6)
              logging.info(f"Reciprocal lattice grid size at sample_rate=4: {size1} \n")
              # create an empty map grid
              data_to_map = gemmi.Ccp4Map()
              # turn MTZ file into map using a sample_rate=4; minimal grid size is
              # placed in relation to the resolution, dmin/sample_rate; sample_rate=4
              # doubles the original grid size
              data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT',
                                                               sample_rate=6)#was 4
              data_to_map.update_ccp4_header(2, True)
            except Exception:
              logging.error(f"Could not create map from {mtz_path}")
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
            try:
              final = os.path.join(output_dir, target+"_"+homo+".ccp4")
              data_to_map.write_ccp4_map(final)
            except Exception:
              logging.error(f"Could not write final map {final}")

      
            1/0


    struct_dir = Path(os.path.join(mtz_dir, struct))
    homo_lst = [homo.stem for homo in struct_dir.iterdir()] 
    for homo in homo_lst:
      homo_dir = os.path.join(struct_dir, homo)
      logging.info(
        f"Converting results for structure {struct}, {mtz_structs.index(struct)+1}/{len(mtz_structs)}")
      if mtz_file in os.listdir(homo_dir):
        logging.info(
          f"Collecting info for {homo}, {homo_lst.index(homo)+1}/{len(homo_lst)}")
        homo_mtz = Path(os.path.join(homo_dir, mtz_file))

        try:
          homo_mtz = Path(os.path.join(homo_dir, mtz_file))
          assert homo_mtz.exists()
        except Exception:
          logging.error(f"Could not find homologue phased MTZ file {homo_mtz}")
          raise

        try:
          data = gemmi.read_mtz_file(str(homo_mtz))
          cell = data.cell
          sg = data.spacegroup
        except Exception:
          logging.error(f"Could not read {homo_mtz}")
         # raise
          pass

        temp_out_file = os.path.join(output_dir, "temp_"+struct+"_"+homo+".ccp4")
        try:
          data_to_map = gemmi.Ccp4Map()
          print("Grid of MTZ file", data_to_map.grid)

          #map_obs = mtz.transform_f_phi_to_map('FWT','PHWT', sample_rate=opt['sample_rate'])
          data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=4)
#           shape = [round(a/1.2/2)*2 for a in data.cell.parameters[:3]]
#           data_to_map.grid = data.transform_f_phi_to_map('FWT', 'PHWT', exact_size=shape)
          print("Grid after converting MTZ to MAP", data_to_map.grid)

          data_to_map.update_ccp4_header(2, True)
          data_to_map.write_ccp4_map(temp_out_file) 
        except Exception:
          logging.error(f"Could not create map from {homo_mtz}")
          raise

        try: 
          # opening temporary map file which shouldn't be neccessary to be written out
          map_to_map = gemmi.read_ccp4_map(temp_out_file)
          map_to_map.setup()

          print("Grid after loading temp file", map_to_map.grid)
          #this bit here expands the unit cell to be 200A^3;
          #Can I expand the unit cell to standard volume and then extract a
          #grid cube (200, 200, 200)
#           xyz_limits = [200, 200, 200]
#           xyz_limits = [100, 100, 100]
          xyz_limits = [50, 50, 50]
          upper_limit = gemmi.Position(*xyz_limits)
          box = gemmi.FractionalBox()
          box.minimum = gemmi.Fractional(0, 0, 0)
          box.maximum = map_to_map.grid.unit_cell.fractionalize(upper_limit)
#           box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(200, 200, 200))
#           box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(100, 100, 100))
          box.maximum = map_to_map.grid.point_to_fractional(map_to_map.grid.get_point(50, 50, 50))
          box.add_margin(1e-5)
          map_to_map.set_extent(box)

          print("Grid after setting XYZ limits for MAP", map_to_map.grid)

          #create a grid with extend x=0-->200, y=0-->200, z=0-->200
          #currently problems as the 200 limit not always reached for all axes;
          #adding a margin maybe that will help
         # new_map.setup()
          # box1 = gemmi.FractionalBox()
         # box1.minimum = gemmi.Fractional(0, 0, 0)
         # box1.maximum = new_map.grid.point_to_fractional(new_map.grid.get_point(200, 200, 200))
         # map_to_map.setup()
         # new_map.set_extent(box1)

          # print("Grid after setting grid dimensions", new_map.grid)

        except Exception:
          logging.error(f"Could not expand map {map_to_map}")          
          raise
#
#          try:
#            map_to_map = gemmi.read_ccp4_map(temp_out_file)
#            map_to_map.setup()
#            print(map_to_map.grid)
#            grid = map_to_map.grid
#            print(grid)
#            new_grid = grid.set_value(200, 200, 200, 4.0)
#            print(new_grid.get_value)
#            xyz_limits = [200, 200, 200]
#            upper_limit = gemmi.Position(*xyz_limits)
#            box = gemmi.FractionalBox()
#            box.minimum = gemmi.Fractional(0, 0, 0)
#            box.maximum = map_to_map.grid.unit_cell.fractionalize(upper_limit)
#            map_to_map.set_extent(box)
#          except Exception:
#            logging.error(f"Could not expand map {map_to_map}")
#            raise



        mtz_state = str(mtz_file).strip(".mtz")
        final_name = struct+"_"+homo+"_"+mtz_state+".ccp4"
        final = os.path.join(output_dir, final_name)
#         final = os.path.join(output_dir, struct+"_"+homo+"_"+mtz_state+".ccp4")
        try:
          map_to_map.write_ccp4_map(final)
#         data_to_map.write_ccp4_map(final)
        except Exception:
          logging.error(f"Could not write final map {final}")
              
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
        "db_file": args.db_file,
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
        "db_file", type=str, help="database file with training lables"
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
        parameters["maps_list"],
            parameters["xyz_limits"],
            parameters["output_dir"],
        )
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data")
        
        
        
        
#peakheight_obs = map_obs.get_nearest_point(cra.atom.pos).value
#peakheight_obs = map_obs.interpolate_value(cra.atom.pos)
#peakheight_obs = map_obs.tricubic_interpolation(cra.atom.pos)
