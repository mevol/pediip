"""
Pipeline for training models with cross validation, recording parameters and performing evaluation.

Designed to make it easier to create and evaluate models with different architectures with the
same training parameters.
"""

# Necessary to make the run as consistent as possible
from numpy.random import seed

seed(1)
#from tensorflow import set_random_seed
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)

import logging
import sqlite3
import re
from pathlib import Path
import os
import shutil
from datetime import datetime
from typing import Callable

import mrcfile
import configargparse
import pandas
import yaml
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
#from modules.cnn.evaluate_model_3d import evaluate
from modules.cnn.training_models.data_generator import DataGenerator

#MAP_DIM = (201, 201, 201)
MAP_DIM = (51, 51, 51)

logging.basicConfig(level=logging.INFO, filename="training.log", filemode="w")


def pipeline(create_model: Callable[[int, int, int, int], Model], parameters_dict: dict):
    """
    Execute the pipeline on the model provided.

    Reads all files in from the training directory path provided, gets their labels from the *ai_labels*
    table in the database provided.

    Sets up Keras ImageDataGenerator for training images with scaling and extra parameters provided,
    and for validation images with scaling only.

    Randomly mixes training data and creates k folds.

    If test directory is provided, evaluates against test data and records that in evaluation folder.

    Records in output directory the history and saves model for each run.

    Parameters in **parameters_dict**:

    - *training_dir* (required) - directory with training images
    - *database_file* (required) - path to database with ai_labels table to get labels from
    - *output_dir* (required) - directory to output files to (this name will be appended with date and time when the training was started)
    - *epochs* (required) - how many epochs to use in each run
    - *batch_size* (required) - size of batch when loading files during training (usually exact multiple of number of files)
    - *test_dir* - directory with testing images
    - *rgb* - whether the model is expecting a 3 channel image
    - image_augmentation_dict - dictionary of key-value pairs to pass as parameters to the Keras ImageGenerator for training images

    :param create_model: function which returns new Keras model to train and evaluate
    :param parameters_dict: dictionary of parameters for use in pipeline
    """

    # Create an output directory if it doesn't exist
    output_dir_path = Path(
        parameters_dict["output_dir"] + "_" + datetime.now().strftime("%Y%m%d_%H%M")
    )
    histories_path = output_dir_path / "histories"
    models_path = output_dir_path / "models"
    evaluations_path = output_dir_path / "evaluations"

    if not output_dir_path.exists():
        # Make one
        try:
            # Make directories
            os.mkdir(output_dir_path)
            os.mkdir(histories_path)
            os.mkdir(models_path)
            os.mkdir(evaluations_path)
            logging.info(f"Created output directories at {output_dir_path}")
        except Exception:
            logging.exception(
                f"Could not create directory at {output_dir_path}.\n"
                f"Please check permissions and location."
            )
            raise

    # Log parameters
    logging.info(f"Running with parameters: {parameters_dict}")

    # Log the key information about the model and run
    with open(output_dir_path / "parameters.yaml", "w") as f:
        yaml.dump(parameters_dict, f)

    # Load training files
    training_dir_path = Path(parameters_dict["training_dir"])
    assert (
        training_dir_path.exists()
    ), f"Could not find directory at {training_dir_path}"
    train_files = [str(file) for file in training_dir_path.iterdir()]
    assert len(train_files) > 0, f"Found no files in {training_dir_path}"
    logging.info(f"Found {len(train_files)} files for training")


    # Load data CSV file with filenames and labels
    data = pandas.read_csv(parameters_dict["sample_lable_lst"])
   # print(data.head)

    X = data['filename']
   # print("dataframe for X", X.head)
    print("dataframe for X", X.shape)
    y = data['ai_lable']
   # print("dataframe for y", y.head)
    print("dataframe for y", y.shape)
    print("Classes and their frequency in the data", data.groupby(y).size())

    #one-hot encoding before creating batches with the datagenerator
    #encode class values as integers
    #encoder = LabelEncoder()
    #encoder.fit(y)
    #encoded_y = encoder.transform(y)
    #dummy_y = np_utils.to_categorical(encoded_y, num_classes=4)
    #print(dummy_y)#a numpy.ndarray


   # X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 100, stratify = y)

   # print("X_train", X_train.head)
   # print("X_train size", X_train.shape)
   # print("X_test", X_test.head)
   # print("X_test size", X_test.shape)
   # print("y_train", y_train.head)
   # print("y_train size", y_train.shape)
   # print("y_test", y_test.head)
   # print("y_test size", y_test.shape)


    # Prepare data generators to get data out
    # Build model
    if parameters_dict["rgb"] is True:
        logging.info("Using 3 channel image input to model")
#        input_shape = (201, 201, 201, 3) #3D
        input_shape = (51, 51, 51, 3) #3D
        color_mode = "rgb"
    else:
        logging.info("Using single channel image input to model")
#        input_shape = (201, 201, 201, 1) #3D
        input_shape = (51, 51, 51, 1) #3D
        color_mode = "grayscale"


    # Model run parameters
    epochs = parameters_dict["epochs"]
    batch_size = parameters_dict["batch_size"]
    print("Number of epochs: ", epochs)
    print("Batch size:", batch_size)

    # New model
    print("Using the following input parameters: ", input_shape)
    model = create_model(input_shape)
    model_info = model.get_config()
    model_architecture = model.summary()
    print(model_architecture)
    logging.info(f"The model architecture is as follows: {model_architecture}")

    #load the training data in batches using the generator;
    #use X and y from training CSV
    training_generator = DataGenerator(X,
                                       y,
                                       dim=MAP_DIM,
                                       batch_size=batch_size,
                                       n_classes=4,
                                       shuffle=True)
    #load the validation data in batches using the generator;
    #use X and y from validation CSV
#    testing_generator = DataGenerator(X_test,
#                                      y_test,
#                                      dim=MAP_DIM,
#                                      batch_size=batch_size,
#                                      n_classes=4,
#                                      shuffle=True)


    #TO DO: need to find a way to run k-fold cross-validation during training


    history = model.fit(
        training_generator,
        steps_per_epoch=int((len(X) / batch_size)),
        epochs=epochs,
        use_multiprocessing=True,
        workers=8)


    # Send history to csv
    history_to_csv(history, histories_path / f"history.csv")
    figure_from_csv(os.path.join(histories_path, "history.csv"),
                    histories_path / f"history.png")
    # Save model as h5
    model.save(str(models_path / f"model.h5"))

    # Make evaluation folder
    if parameters_dict["test_dir"]:
        logging.info("Performing evaluation of model")
        evaluation_dir_path = str(evaluations_path / f"evaluation")
        if not Path(evaluation_dir_path).exists():
            os.mkdir(evaluation_dir_path)
        evaluate(
            str(models_path / f"model.h5"),
            parameters_dict["sample_lable_lst"],#need to replace with validation CSV filenames
           #parameters_dict["database_file"],#need to replace with validation CSV lables
            evaluation_dir_path,
            rgb=parameters_dict["rgb"],
            )
    else:
        logging.info(
           "Requires test directory for evaluation. "
            "No evaluation performed"
            )

    # Load the model config information as a yaml file
    with open(output_dir_path / "model_info.yaml", "w") as f:
        yaml.dump(model_info, f)

    # Try to copy log file if it was created in training.log
    try:
        shutil.copy("training.log", output_dir_path)
    except FileExistsError:
        logging.warning("Could not find training.log to copy")
    except Exception:
        logging.warning("Could not copy training.log to output directory")


def pipeline_from_command_line(
    create_model: Callable[[int, int, int, int], Model], rgb: bool = False #3D
):
    """
    Run the training pipeline from the command line with config file

    Get parameters from the command line and pass them to training_pipeline in the parameter dict

    :param create_model: function which returns new Keras model to train and evaluate
    :param rgb: whether the model is expecting a 3 channel image
    """

    # Get from pipeline
    argument_dict = get_pipeline_parameters()


    # Add rgb parameter
    assert isinstance(
        rgb, bool
    ), f"Must provide bool for rgb, got {type(rgb)} of value {rgb}"
    argument_dict["rgb"] = rgb

    logging.info(f"Running model with parameters: {argument_dict}")

    # Send parameters to full pipeline
    pipeline(create_model, argument_dict)


def get_pipeline_parameters() -> dict:
    """
    Extract the parameters from a mixture of config file and command line using configargparse

    :return parameters_dict: dictionary containing all parameters necessary for pipeline
    """

    # Set up parser to work with command line argument or yaml file
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="Training pipeline for a Keras model which is parameterized from the command line "
        "or YAML config file.\n"
        "To perform image augmentation, please provide a dictionary in the config file entitled "
        "image_augmentation_dict with kay value pair matching the parameters listed here: https://keras.io/preprocessing/image/"
        " The images will automatically be rescaled.",
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        is_config_file=True,
        help="config file to specify the following parameters in",
    )
    parser.add_argument(
        "--training_dir",
        required=True,
        help="directory with training images in"
    )

    parser.add_argument(
        "--sample_lable_lst",
        required=True,
        help="list of sample files with associated training lables",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory to output results files to. Will be appended with date and time of program run",
    )
    parser.add_argument(
        "--epochs", type=int,
        required=True,
        help="number of epochs to use in each run"
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="size of batch to load during training and validation. Should be exact factor of the number of images provided",
    )

    (known_args, unknown_args) = parser.parse_known_args()

    argument_dict = vars(known_args)

    # Try to extract image_augmentation_dict
    try:
        assert unknown_args
        assert "--image_augmentation_dict" in unknown_args
        # Extract values from config file
        with open(known_args.config, "r") as f:
            image_augmentation_dict = yaml.load(f.read())["image_augmentation_dict"]
    except (KeyError, AssertionError):
        logging.warning(
            f"Could not find image_augmentation_dict in {known_args.config}, performing scaling only"
        )
        image_augmentation_dict = {}
    assert isinstance(
        image_augmentation_dict, dict
    ), f"image_augmentation_dict must be provided as a dictionary in YAML, got {image_augmentation_dict}"

    argument_dict["image_augmentation_dict"] = image_augmentation_dict


    return argument_dict


if __name__ == "__main__":

    args = get_pipeline_parameters()
    print(f"Received the following arguments: {args}")
