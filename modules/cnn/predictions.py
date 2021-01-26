"""Module containing useful functions for getting predictions on
images from a model and outputting the predictions in a useful format"""

import json
import logging
from pathlib import Path

import configargparse
import keras.models
import mrcfile
import numpy as np

from topaz3.maps_to_images import slice_map

IMG_DIM = (201, 201)


def predictions_from_images(image_stack: np.ndarray, model_file: str) -> np.ndarray:
    """Get predictions from a model on the image stack provided"""
    try:
        model = keras.models.load_model(model_file)
    except OSError:
        logging.exception(f"Could not find .h5 model at {model_file}")
        raise

    predictions = model.predict(image_stack)

    return predictions


def map_to_images(map_file: str, slices_per_axis: int, rgb: bool = False) -> np.ndarray:
    """Convert a map to an image stack and scale it properly"""
    logging.info(f"Extracting data from {map_file}")
    try:
        with mrcfile.open(map_file) as mrc:
            volume = mrc.data
    except ValueError:
        logging.exception(f"Expected a .map file, not {map}")
        raise
    except FileNotFoundError:
        logging.exception(f"No file found at {map_file}, please provide .map file path")
        raise
    except Exception:
        logging.exception(
            f"Could not get data from {map_file}, please provide .map file path"
        )
        raise

    # Get image slices
    logging.info(f"Slicing map into {slices_per_axis} images on each axis")
    image_stack = slice_map(volume, slices_per_axis)

    # Check dimensions are correct
    assert (
        image_stack.shape[1],
        image_stack.shape[2],
    ) == IMG_DIM, f"Expected image slices of {IMG_DIM}, not {(image_stack.shape[1], image_stack.shape[2])}"

    logging.info(f"Got {image_stack.shape[0]} slices for prediction")

    # Scale slices for input to neural network
    for slice_num in range(image_stack.shape[0]):
        # Get slice
        slice = image_stack[slice_num, :, :]
        # Scale slice
        slice = (slice - slice.min()) / (slice.max() - slice.min())

        # Return to image_stack (in place)
        image_stack[slice_num, :, :] = slice

    if rgb:
        # Turn into RGB image array
        image_stack_rgb = np.stack((image_stack,) * 3, axis=3)
        return image_stack_rgb
    else:
        # Add a 4th dimension for the benefit of keras and return
        return np.expand_dims(image_stack, 3)


def predictions_from_map(
    map_file: str, slices_per_axis: int, model_file: str, rgb: bool = False
) -> np.ndarray:
    """
    Generate image slices from map file and get predictions from the model for each slice.

    Return the raw image predictions as a numpy array.

    :param map_file: map file to get predictions from
    :param slices_per_axis: number of image slices to take along each axis to generate predictions
    :param model_file: .h5 file to load Keras model from
    :param rgb: indicate whether model expects 3 channel image
    :returns: numpy array with predictions for each image slice
    """
    image_stack = map_to_images(map_file, slices_per_axis, rgb)

    # Get predictions
    predictions = predictions_from_images(image_stack, model_file)

    return predictions


def predict_original_inverse(
    original_map_file: str,
    inverse_map_file: str,
    slices_per_axis: int,
    model_file: str,
    output_dir: str = None,
    raw_pred_filename: str = "raw_predictions.json",
    average_pred_filename: str = "avg_predictions.json",
    rgb: bool = False,
) -> np.ndarray:
    """
    Get predictions for the original and inverse maps at the same time and output results.

    Slices both map files into 2d images, loads model from .h5 file and gets predictions on the image
    slices.
    These predictions are saved as *raw_predictions.json*, by default, in the **output_dir**.

    The predictions are then averaged across each structure and these values are returned.
    They are also saved in *avg_predictions.json*, by default.

    :param original_map_file: map file for original hand
    :param inverse_map_file: map file for inverse hand
    :param slices_per_axis: number of image slices to take along each axis to generate predictions
    :param model_file: .h5 file to load Keras model from
    :param output_dir: directory to output results files to
    :param rgb: whether the model is expecting a 3 channel image
    :param raw_pred_filename: set the filename for the raw predictions json output
    :param average_pred_filename: set the filename for the average predictions json output
    :returns: numpy array with averaged predictions
    """
    logging.info("Getting predictions for original and inverse maps pair")
    logging.info(f"Original at: {original_map_file}")
    logging.info(f"Inverse at: {inverse_map_file}")

    # Get image stacks
    original_image_stack = map_to_images(original_map_file, slices_per_axis, rgb=rgb)
    inverse_image_stack = map_to_images(inverse_map_file, slices_per_axis, rgb=rgb)
    # Add image stacks together with original first, should have shape
    # of (6*slices_per_axis, 201, 201, 1) for easy input to neural network
    total_image_stack = np.concatenate(
        (original_image_stack, inverse_image_stack), axis=0
    )

    # Get predictions
    logging.info(f"Getting predictions from model at {model_file}")
    predictions = predictions_from_images(total_image_stack, model_file)

    # Record raw predictions
    if output_dir:
        assert Path(
            output_dir
        ).is_dir(), f"Could not find expected directory at {output_dir}"
    # Split the predictions in half to match the original and inverse pairs
    raw_predictions = {
        "Original": predictions[: int(len(predictions) / 2)].tolist(),
        "Inverse": predictions[int(len(predictions) / 2) :].tolist(),
    }
    try:
        if output_dir:
            with open(Path(output_dir) / raw_pred_filename, "w") as raw_pred_file:
                json.dump(raw_predictions, raw_pred_file, indent=4)
        else:
            print(f"Raw predictions:\n{json.dumps(raw_predictions, indent=2)}")
    except Exception:
        logging.exception(f"Could not write raw predictions to {raw_pred_file.name}")
        raise

    # Record the average predictions
    avg_predictions = {
        "Original": {
            0: np.mean([pred[0] for pred in raw_predictions["Original"]]),
            1: np.mean([pred[1] for pred in raw_predictions["Original"]]),
        },
        "Inverse": {
            0: np.mean([pred[0] for pred in raw_predictions["Inverse"]]),
            1: np.mean([pred[1] for pred in raw_predictions["Inverse"]]),
        },
    }
    try:
        if output_dir:
            with open(Path(output_dir) / average_pred_filename, "w") as avg_pred_file:
                json.dump(avg_predictions, avg_pred_file, indent=4)
        else:
            print(f"Averaged predictions:\n{json.dumps(avg_predictions, indent=2)}")
    except Exception:
        logging.exception(
            f"Could not write average predictions to {avg_pred_file.name}"
        )
        raise

    return avg_predictions


def command_line():
    """
    Command line wrapper for the predict_original_inverse function
    """
    logging.basicConfig(level=logging.INFO)

    # Set up parser to work with command line argument or yaml file
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="Generate predictions from original and inverse hand map files using the model provided. "
        "If an output directory is provided, the raw and average predictions will be saved there. "
        "Otherwise they will be printed to the terminal.",
    )

    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file to specify the following options in",
    )
    parser.add_argument(
        "--original_map_file", required=True, help="map file of original hand"
    )
    parser.add_argument(
        "--inverse_map_file", required=True, help="map file of inverse hand"
    )
    parser.add_argument(
        "--slices_per_axis",
        required=True,
        type=int,
        help="slices to be taken of map per axis for predictions",
    )
    parser.add_argument(
        "--model_file", required=True, help=".h5 file to load model from"
    )
    parser.add_argument(
        "--output_dir", default=None, help="directory to store results in"
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        default=False,
        help="include this option if using a model which expects 3d images",
    )

    args = parser.parse_args()

    # Print out the parameters this is being run on

    logging.info(
        f"Prediction called with following parameters:\n"
        f"Original map file: {args.original_map_file}\n"
        f"Inverse map file: {args.inverse_map_file}\n"
        f"Sliced with {args.slices_per_axis}\n"
        f"Predictions using model at {args.model_file}\n"
        f"Output to: {args.output_dir}\n"
        f"Using rgb images: {args.rgb}"
    )

    predictions = predict_original_inverse(
        args.original_map_file,
        args.inverse_map_file,
        args.slices_per_axis,
        args.model_file,
        args.output_dir,
        rgb=args.rgb,
    )

    return predictions


if __name__ == "__main__":
    command_line()
