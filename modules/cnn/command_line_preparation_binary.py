"""Command line tool to enable entire image preparation step from yaml config file"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from modules.cnn.prepare_training_data_binary import prepare_training_data_binary

example_config = """maps_list: /path/to/maps/list
xyz_limits:
  - 200
  - 200
  - 200
output_dir: /output/image/slices/directory
"""

def params_from_yaml(args):
    """Extract the parameters for preparation from a yaml file and return a dict"""
    # Check the path exists
    try:
        config_file_path = Path(args.config)
        assert config_file_path.exists()
    except Exception:
        logging.error(f"Could not find config file at {args.config}")
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

    if "verbose" not in params.keys():
        params["verbose"] = True

    return params


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Transform a large directory of cells with phase information "
        "into labelled image slices ready for AI training.\n "
        "See https://github.com/TimGuiteDiamond/topaz3 for further details"
    )

    parser.add_argument(
        "config", help="yaml file which contains parameters for data preparation"
    )
    parser.add_argument(
        "--make-output",
        action="store_true",
        help="automatically generate maps and images directories and database for output",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="creates example in the file provided as *config* option",
    )

    args = parser.parse_args()

    if args.example:
        logging.info(f"Creating example in {args.config}")
        with open(args.config, "w") as cf:
            cf.write(example_config)
        sys.exit(0)

    logging.info(f"Extracting parameters from {args.config}")
    parameters = params_from_yaml(args)
    
    print(parameters)

    if args.make_output:
        logging.info("Generating output directories")
        try:
            os.mkdir(parameters["output_dir"])
        except TypeError:
            logging.error(
                f"Expected file path for output_dir, got {parameters['output_dir']}"
            )
            raise
        except FileExistsError:
            logging.error(f"Using existing output_dir at {parameters['output_dir']}")
        except PermissionError:
            logging.error(
                f"Do not have permission to create output_dir at {parameters['output_dir']}"
            )
            raise


    logging.info(f"Converting mtz files to map files")
    prepare_training_data_binary(
        parameters["maps_list"],
        parameters["xyz_limits"],
        parameters["output_dir"],
    )



if __name__ == "__main__":

    main()
