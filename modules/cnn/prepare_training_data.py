import argparse
import logging
import os
import yaml
import gemmi
import csv
import sqlite3
#import multiprocessing

from pathlib import Path
from typing import List

from modules.cnn.delete_temp_files import delete_temp_files

def prepare_training_data(
    mtz_directory: str,
    mtz_file: str,
    xyz_limits: List[int],
    output_directory: str,
    db_file: str,
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

    # Innitialise connection to database
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
    except Exception:
        logging.error(
            f"Could not connect to database at {db_file}"
        )
        raise

    if not os.path.exists(os.path.join(output_dir, "conv_map_list.csv")):
      with open(os.path.join(output_dir, "conv_map_list.csv"), "w") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["filename", "ai_lable"])
      

    # Get lists of child directories
    mtz_structs = [struct.stem for struct in mtz_dir.iterdir()]
    mtz_structs = sorted(mtz_structs)
    logging.debug(f"Following structures found to transform: {mtz_structs}")

#this below works but runs serial
    for struct in mtz_structs:
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
            raise
          temp_out_file = os.path.join(output_dir, "temp_"+struct+"_"+homo+".ccp4")
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
            map_to_map.setup()
            xyz_limits = [200, 200, 200]
            upper_limit = gemmi.Position(*xyz_limits)
            box = gemmi.FractionalBox()
            box.minimum = gemmi.Fractional(0, 0, 0)
            box.maximum = map_to_map.grid.unit_cell.fractionalize(upper_limit)
            map_to_map.set_extent(box)
          except Exception:
            logging.error(f"Could not expand map {map_to_map}")          
            raise
          mtz_state = str(mtz_file).strip(".mtz")
          final = os.path.join(output_dir, struct+"_"+homo+"_"+mtz_state+".ccp4")
          try:
            map_to_map.write_ccp4_map(final)
            cur.execute('''
                         SELECT refinement_success_lable, homologue_name_id
                         FROM homologue_stats
                         INNER JOIN homologue_name
                         ON homologue_name.id = homologue_stats.homologue_name_id
                         INNER JOIN pdb_id
                         ON pdb_id.id = homologue_name.pdb_id_id
                         WHERE homologue_name = "%s"
                         AND pdb_id.pdb_id = "%s"
                         '''%(homo, struct))
            lable = (cur.fetchone())[0]
            print(lable)
            if lable == "1a":
              new_lable = 1
            if lable == "1b":
              new_lable = 2
            if lable == "2a":
              new_lable = 3
            if lable == "2b":
              new_lable = 4
            print(new_lable) 
            with open(os.path.join(output_dir, "conv_map_list.csv"), "a", newline = "") as out_csv:
              writer = csv.writer(out_csv)
              writer.writerow([final, new_lable])
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
            parameters["mtz_dir"],
            parameters["mtz_file"],
            parameters["xyz_limits"],
            parameters["maps_dir"],
            parameters["db_file"],
            parameters["delete_temp"],
        )
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data")
