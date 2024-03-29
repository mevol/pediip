import argparse
import logging
import os
import yaml
import gemmi
import csv
import re

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from sys import getsizeof
from scipy import ndimage


def slice_map(volume, slices_per_axis, augmentation):
    """Slice the volume into 2d panes along x, y, z axes and return as an image stack"""
    # Check volume is equal in all directions
    assert (
        volume.shape[0] == volume.shape[1] == volume.shape[2]
    ), f"Please provide a volume which has dimensions of equal length, not {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}"

    length = volume.shape[0]

    # randomly select slices at each axis; the number of picks is determined by
    # slices_per_axis; iterate over each axis; reshape to same dimensions as for first
    # axis; stack vertically
    # do data augmentation as rotation for a random angle between 0 and 90 deg
    # for all even numbers in the total image stack
    # check that the remainder of division is 0 and hence the result even

    first_pick = np.random.choice(length, slices_per_axis, replace=False)
    second_pick = np.random.choice(length, slices_per_axis, replace=False)
    third_pick = np.random.choice(length, slices_per_axis, replace=False)

    image_stack = np.zeros((slices_per_axis * 3, length, length))

    index = 0
    for s in first_pick:
        deg1 = np.random.choice(90, 1, replace=False)[0]
        stack1 = np.copy(volume[s, :, :])
        if augmentation == True:
          rotate1 = ndimage.rotate(stack1, angle = deg1, reshape=False)
          image_stack[index, :, :] = rotate1
        else:
          image_stack[index, :, :] = stack1
        index = index + 1

    for ss in second_pick:
        deg2 = np.random.choice(90, 1, replace=False)[0]
        stack2 = np.copy(volume[:, ss, :])
        if augmentation == True:
          rotate2 = ndimage.rotate(stack2, angle = deg2, reshape=False)
          image_stack[index, :, :] = rotate2
        else:
          image_stack[index, :, :] = stack2
        index = index + 1

    for sss in third_pick:
        deg3 = np.random.choice(90, 1, replace=False)[0]
        stack3 = np.copy(volume[:, :, sss])
        if augmentation == True:
          rotate3 = ndimage.rotate(stack3, angle = deg3, reshape=False)
          image_stack[index, :, :] = rotate3
        else:
          image_stack[index, :, :] = stack3
        index = index + 1

    byte_size_stack = getsizeof(image_stack)

    return image_stack, byte_size_stack

def prepare_training_data_random_pick_combined(
    maps_list: str,
    xyz_limits: List[int],
    slices_per_axis: int,
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
                                                               sample_rate=6)#was 4
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
            logging.info(f"Size of standardise map when finished: {map_array.shape} \n")
        except Exception:
            logging.error(f"Could not expand map {map_file_path} \n")
            raise

        try:
            # create a new array to hold the scaled, rounded and augmented images
            length = int(xyz_limits[0])+1
            edited_image_slices = np.zeros((slices_per_axis * 3,
                                            length,
                                            length))
            # Slice the volume into images
            image_slices, bytes = slice_map(map_array, slices_per_axis, augmentation)
            # Iterate through images, scale them and save them in output_directory
            for slice_num in range(image_slices.shape[0]):
                # Get slice
                slice = image_slices[slice_num, :, :]
                # Scale slice
                slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
                # Round to the nearest integer
                slice_scaled_int = np.rint(slice_scaled)
                # combine the slices to a new image stack for training
                edited_image_slices[slice_num, :, :] = slice_scaled_int#volume[

            # check the number of edited image slices
            assert len(edited_image_slices) == 60
            total_bytes = total_bytes + bytes
            number_maps = number_maps + 1
            logging.info(f"Accumulated byte size: {total_bytes} \n")
            logging.info(f"Total number of maps processed: {number_maps} \n")

        except Exception:
            raise
    except Exception:
        logging.error(f"Could not open input map list \n")
        raise
    return edited_image_slices



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
              "xyz_limits": args.xyz_limits,
              "slices": args.slices_per_axis}

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
    cmd_parser.add_argument(
        "slices",
        type=int,
        help="number of image slices to produce per axis, default=20",
        default=20)
    cmd_parser.set_defaults(func=params_from_cmd)

    # Extract the parameters based on the yaml/command line argument
    args = parser.parse_args()
    parameters = args.func(args)

    # Execute the command
    try:
        prepare_training_data_binary(
            parameters["maps_list"],
            parameters["xyz_limits"],
            parameters["slices"])
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data \n")
