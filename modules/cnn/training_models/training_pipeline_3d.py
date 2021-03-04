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
import tensorflow

import mrcfile
import configargparse
import pandas
import yaml
import numpy as np
import math
#from keras import Model
from tensorflow.keras import Model
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
#from modules.cnn.evaluate_model_3d import evaluate
from modules.cnn.training_models.data_generator import DataGenerator

#MAP_DIM = (201, 201, 201)
MAP_DIM = (101, 101, 101)
#MAP_DIM = (51, 51, 51)

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

    X = data['filename']
    y = data['ai_lable']
    print("Classes and their frequency in the data", data.groupby(y).size())
    class_frequency = data.groupby(y).size()
    logging.info(f"Number of samples per class {class_frequency}")    


    label_dict = y.to_dict()
 #   print(label_dict)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify = y)

    print("Number of samples in y_test ", len(y_test))
    print("Number of samples in X_test ", len(X_test))

    
    partition = {"train" : X_train,
                 "validate" : X_test}

#    print(partition["train"])
#    print(partition["validate"])

    print("Length of partition validate: ", len(partition["validate"]))

    # Prepare data generators to get data out
    # Build model
    if parameters_dict["rgb"] is True:
        logging.info("Using 3 channel image input to model")
#        input_shape = (201, 201, 201, 3) #3D
        input_shape = (101, 101, 101, 3) #3D
#        input_shape = (51, 51, 51, 3) #3D
        color_mode = "rgb"
    else:
        logging.info("Using single channel image input to model")
#        input_shape = (201, 201, 201, 1) #3D
        input_shape = (101, 101, 101, 1) #3D
#        input_shape = (51, 51, 51, 1) #3D
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


    training_generator = DataGenerator(partition["train"],#X
                                       label_dict,#y
                                       dim=MAP_DIM,
                                       batch_size=batch_size,
                                       n_classes=4,
                                       shuffle=True)


    testing_generator = DataGenerator(partition["validate"],
                                      label_dict,
                                      dim=MAP_DIM,
                                      batch_size=batch_size,
                                      n_classes=4,
                                      shuffle=False)#was True


    #TO DO: need to find a way to run k-fold cross-validation during training


    history = model.fit(
        training_generator,
        steps_per_epoch=int((len(X_train) / batch_size)),#len(X) if not using train-test-split
        epochs=epochs,
        validation_data=testing_generator,
        validation_steps=(len(X_test) / batch_size),
        use_multiprocessing=True,
        workers=8)#8)


    # Send history to csv
    history_to_csv(history, histories_path / f"history.csv")
    figure_from_csv(os.path.join(histories_path, "history.csv"),
                    histories_path / f"history.png")
    # Save model as h5
    model.save(str(models_path / f"model.h5"))




    # Make evaluation folder

    logging.info("Performing evaluation of model")
    evaluation_dir_path = str(evaluations_path / f"evaluation")
    if not Path(evaluation_dir_path).exists():
      os.mkdir(evaluation_dir_path)

    logging.info("Getting predictions")
    
    print(int(math.ceil(len(X_test) / batch_size)))
    print(int(np.round(len(X_test) / batch_size)))

    try:
      predictions = model.predict(
                          testing_generator,
                          steps=int(math.ceil(len(X_test) / batch_size)),
                          verbose=1)
                          
      print("Length of predictions: ", len(predictions))                    
    except ValueError:
      logging.exception(
              "Ensure the RGB option is set correctly for your model - "
              "Some models expect 3 channel data")
      raise

    try:
      preds_rounded = np.round(predictions, 0)
      print(preds_rounded)
      print("Length of predictions rounded: ", len(preds_rounded))
    except Exception:
      logging.warning("Could not round predictions")
      raise

    try:
      classification_metrics = metrics.classification_report(y_test, preds_rounded)
      print(classification_metrics)
    except Exception:
      logging.warning("Could not get multi-class classification report")
      raise






#    # Per map analysis
#    predictions_1 = [x for x in predictions if x[1] > x[0]]
#    predictions_0 = [x for x in predictions if x[1] < x[0]]
#    logging.info(f"Predicted good value {len(predictions_1)} times")
#    logging.info(f"Predicted bad value {len(predictions_0)} times")
#
#    predictions_decoded = [int(pred[1] > pred[0]) for pred in predictions]
#
#    # Save raw predictions
#    raw_dataframe = pandas.DataFrame(
#        {
#            "File": testing_dataframe["Files"],
#            "0": predictions[:, 0],
#            "1": predictions[:, 1],
#            "True Score": test_labels,
#        }
#    )
#    raw_dataframe.set_index("File", inplace=True)
#    raw_dataframe.to_csv(output_dir_path / "raw_predictions.csv")
#
#    logging.info("Per map analysis:")
#    per_map_class = classification_report(
#        predictions_decoded, testing_dataframe["Labels"], output_dict=True
#    )
#    per_map_class_frame = pandas.DataFrame(per_map_class).transpose()
#    per_map_conf = confusion_matrix(predictions_decoded, testing_dataframe["Labels"])
#    per_map_conff_frame = pandas.DataFrame(per_map_conf)
#    logging.info(per_map_class)
#    logging.info(per_map_conf)
#    # Save reports to csv
#    per_map_class_frame.to_csv(output_dir_path / "per_map_class.csv")
#    per_map_conff_frame.to_csv(output_dir_path / "per_map_conf.csv")
#    logging.info("Per map analysis complete")
#
    logging.info("Evaluations complete.")


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
