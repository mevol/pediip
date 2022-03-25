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
from PIL import Image
from sys import getsizeof


def slice_map(volume, slices_per_axis):
    """Slice the volume into 2d panes along x, y, z axes and return as an image stack"""



    # Check volume is equal in all directions
    assert (
        volume.shape[0] == volume.shape[1] == volume.shape[2]
    ), f"Please provide a volume which has dimensions of equal length, not {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}"

    length = volume.shape[0]

    random_pick = np.random.choice(length, size = slices_per_axis)

    # Array to return the images
#    image_stack = np.zeros((slices_per_axis * 3, length, length))

###### RANDOMLY PICK 20 SLICES

    
    print(random_pick)

    print(range(slices_per_axis))

    # Get x slices and put in image_stack
#    for slice in range(slices_per_axis):
#        image_stack[slice, :, :] = volume[
#            (slice + 1) * int((length) / (slices_per_axis + 1)), :, :
#        ]
#    # random pick try 1
#    for slice in random_pick:
#        temp_stack1 = volume[slice, :, :]
#        print("Shape of random slice axis X: ", temp_stack1.shape)
#        image_stack = np.vstack([*image_stack, temp_stack1[None]])
#        print("Shape of image stack axis X: ", image_stack.shape)

    # Get y slices and put in image_stack
#    for slice in range(slices_per_axis):
#        image_stack[slice + slices_per_axis, :, :] = volume[
#            :, (slice + 1) * int((length) / (slices_per_axis + 1)), :
#        ]
#    # random pick try 1
#    for slice in random_pick:
#        temp_stack2 = volume[:, slice, :]
#        image_stack = np.vstack([*image_stack, temp_stack2[None]])

    # Get z slices and put in image_stack
#    for slice in range(slices_per_axis):
#        image_stack[slice + (slices_per_axis * 2), :, :] = volume[
#            :, :, (slice + 1) * int((length) / (slices_per_axis + 1))
#        ]
#    # random pick try 1
#    for slice in random_pick:
#        temp_stack3 = volume[:, :, slice]
#        image_stack = np.vstack([*image_stack, temp_stack3[None]])


    stack1 = volume[np.random.choice(volume.shape[0], slices_per_axis, replace=False), :, :]
    print("Stack1 shape: ", stack1.shape)
    stack2 = volume[:, np.random.choice(volume.shape[1], slices_per_axis, replace=False), :]
    print("Stack2 shape: ", stack2.shape)
    stack2 = stack2.reshape(slices_per_axis, volume.shape[0], volume.shape[0])
    print("Stack2 shape after reshape: ", stack2.shape)
    stack3 = volume[:, :, np.random.choice(volume.shape[2], slices_per_axis, replace=False)]
    print("Stack3 shape: ", stack3.shape)
    stack3 = stack3.reshape(slices_per_axis, volume.shape[0], volume.shape[0])
    print("Stack3 shape after reshape: ", stack3.shape)

#    image_stack = np.append(image_stack, stack1)
#    image_stack = np.append(image_stack, stack2)
#    image_stack = np.append(image_stack, stack3)

    image_stack = np.vstack([stack1, stack2])
    print("After first addition: ", image_stack.shape)
    image_stack = np.vstack([image_stack, stack3])
    print("After second addition: ", image_stack.shape)
#    image_stack = np.vstack([image_stack, stack3])
#    print("After third addition: ", image_stack.shape)

    byte_size_stack = getsizeof(image_stack)
    
    image_stack = image_stack.reshape(60, 101, 101)

    return image_stack, byte_size_stack


#def TileImage(imgs, picturesPerRow=10):
#    """ Convert to a true list of 16x16 images
#    """
#
#    # Calculate how many columns
#    picturesPerColumn = imgs.shape[0]/picturesPerRow + 1*((imgs.shape[0]%picturesPerRow)!=0)
#
#    # Padding
#    rowPadding = picturesPerRow - imgs.shape[0]%picturesPerRow
#    imgs = vstack([imgs,zeros([rowPadding,imgs.shape[1]])])
#
#    # Reshaping all images
#    imgs = imgs.reshape(imgs.shape[0],100,100)
#
#    # Tiling Loop (The conditionals are not necessary anymore)
#    tiled = []
#    for i in range(0,picturesPerColumn*picturesPerRow,picturesPerRow):
#        tiled.append(hstack(imgs[i:i+picturesPerRow,:,:]))
#
#
#    return vstack(tiled)

def prepare_training_data_random_pick_combined(
    maps_list: str,
    xyz_limits: List[int],
    slices_per_axis: int):
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
        print("DATA for preparation: ", data)
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
            print("Working on target: ", target)
            logging.info(f"Working on target: {target} \n")
        except Exception:
            pass
        try:
            homo = input_map_path.split("/")[12]
            print("Working on homologue: ", homo)
            logging.info(f"Working on homologue: {homo} \n")
        except Exception:
            print("Could not find homologue to work with.")
            logging.error(f"Could not find homologue to work with. \n")
            pass
        # Check path to map exists
        try:
            map_file_path = Path(os.path.realpath(real_path_to_map_opt))
            print(map_file_path)
            assert map_file_path.exists()
        except Exception:
            logging.error(f"Could not find mtz directory at {map_file_path} \n")
            pass

        try:
            # try opening MTZ file
            data = gemmi.read_mtz_file(str(map_file_path))
            # get reciprocal lattice grid size
            recip_grid = data.get_size_for_hkl()
            print("reciprocal lattice grid: ", recip_grid)
            logging.info(f"Original size of reciprocal lattice grid: {recip_grid} \n")
            # get grid size in relation to resolution and a sample rate of 4
            size1 = data.get_size_for_hkl(sample_rate=4)
            logging.info(f"Reciprocal lattice grid size at sample_rate=4: {size1} \n")
            # create an empty map grid
            data_to_map = gemmi.Ccp4Map()
            print("Grid of MTZ file", data_to_map.grid)
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
            print("XYZ limits to make standardised map: ", xyz_limits[0],
                                                           xyz_limits[1],
                                                           xyz_limits[2])
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
            print("Size of standardise map when finished: ", map_array.shape)
            print("Grid after setting XYZ limits for MAP", map_grid)
        except Exception:
            logging.error(f"Could not expand map {map_file_path} \n")
            raise

        try:
            # create a new array to hold the scaled, rounded and augmented images
            length = int(xyz_limits[0])+1
            edited_image_slices = np.zeros((slices_per_axis * 3,
                                            length,
                                            length))
            print("Dimensions of empty edited image stack: ", edited_image_slices.shape)
            # Slice the volume into images
            image_slices, bytes = slice_map(map_array, slices_per_axis)
            # Iterate through images, scale them and save them in output_directory
            print("Number of slices to edit and manipulate: ", len(image_slices))
            print("Dimensions of returned image stack: ", image_slices.shape)
            for slice_num in range(image_slices.shape[0]):
                #print("Working on slice number: ", slice_num)
                # Get slice
                slice = image_slices[slice_num, :, :]
                print("Slice dimension: ", slice.shape)
                # Scale slice
                slice_scaled = ((slice - slice.min()) / (slice.max() - slice.min())) * 255.0
                # Round to the nearest integer
                slice_scaled_int = np.rint(slice_scaled)
                print("Dimensions of scaled image slices: ", slice_scaled_int.shape)
                edited_image_slices[slice_num, :, :] = slice_scaled_int#volume[
                
                # ENTER IMAGE AUGMENTATION HERE
            # check the number of edited image slices
            assert len(edited_image_slices) == 60
            print("The number of edited image slices to be combined is: ",
                          len(edited_image_slices))
            print("Shape of edited image slice stack: ", edited_image_slices.shape)
            total_bytes = total_bytes + bytes
            number_maps = number_maps + 1
            print("Accumulated byte size: ", total_bytes)
            print("Total number of maps processed: ", number_maps)


#                tiled_img = TileImage(edited_image_slices)

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
            f"Could not extract parameters from yaml file at {config_file_path} \n"
        )
        raise

    return params


def params_from_cmd(args):
    """Extract the parameters for prepare_training_data from the command line and return a dict"""
    params = {
        "maps_list": args.maps_list,
        "xyz_limits": args.xyz_limits,
        "slices": args.slices_per_axis,
    }

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
        "slices", type=int, help="number of image slices to produce per axis, default=20",
        default=20
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
            parameters["slices"]
        )
    except KeyError as e:
        logging.error(f"Could not find parameter {e} to prepare training data \n")
