"""
Pipeline for training models with cross validation, recording parameters and performing evaluation.

Designed to make it easier to create and evaluate models with different architectures with the
same training parameters.
"""

# Necessary to make the run as consistent as possible
from numpy.random import seed

seed(1)
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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mrcfile
import configargparse
import pandas
import yaml
import numpy as np
import math
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
#from modules.cnn.evaluate_model_3d import evaluate
from modules.cnn.training_models.data_generator_binary import DataGenerator

print(tensorflow.__version__)

#MAP_DIM = (201, 201, 201)
#MAP_DIM = (101, 101, 101)
#MAP_DIM = (51, 51, 51)

logging.basicConfig(level=logging.INFO, filename="training.log", filemode="w")


def pipeline(create_model: Callable[[int, int, int], Model], parameters_dict: dict):
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

    MAP_DIM = parameters_dict["image_dim"]
    print("map dimensions ", MAP_DIM)

#    # Load training files
#     training_dir_path = Path(parameters_dict["training_dir"])
#     assert (
#         training_dir_path.exists()
#     ), f"Could not find directory at {training_dir_path}"
#     train_files = [str(file) for file in training_dir_path.iterdir()]
#     assert len(train_files) > 0, f"Found no files in {training_dir_path}"
#     logging.info(f"Found {len(train_files)} files for training")
# 
#
#    # Load data CSV file with filenames and labels
#     data = pandas.read_csv(parameters_dict["sample_lable_lst"])
# 
#     X = data['filename']
#     y = data['ai_lable']
#     print("Classes and their frequency in the data", data.groupby(y).size())
#     class_frequency = data.groupby(y).size()
#     logging.info(f"Number of samples per class {class_frequency}")    
# 
# 
#     label_dict = y.to_dict()
#    #print(label_dict)
# 
# 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify = y)
# 
#    #print("Number of samples in y_test ", len(y_test[:-2]))
#    #print("Number of samples in X_test ", len(X_test))
#    #print("Number of samples in X_test ", len(X_test[:-2]))
#     
#     partition = {"train" : X_train,
#                  "validate" : X_test[:-2]}
# 
#    #print("Length of partition train", len(partition["train"]))
#    #print(partition["validate"])
#
#    #print("Length of partition validate: ", len(partition["validate"]))
#
#    # Prepare data generators to get data out
#    # Build model
#    if parameters_dict["rgb"] is True:
#        logging.info("Using 3 channel image input to model")
##        input_shape = (201, 201, 201, 3) #3D
#        input_shape = (101, 101, 101, 3) #3D
##        input_shape = (51, 51, 51, 3) #3D
#        color_mode = "rgb"
#    else:
#        logging.info("Using single channel image input to model")
##        input_shape = (201, 201, 201, 1) #3D
#        input_shape = (101, 101, 101, 1) #3D
##        input_shape = (51, 51, 51, 1) #3D
#        color_mode = "grayscale"
#
#
#    # Model run parameters
#     epochs = parameters_dict["epochs"]
#     batch_size = parameters_dict["batch_size"]
#     print("Number of epochs: ", epochs)
#     print("Batch size:", batch_size)
# 
#    # New model
#     print("Using the following input parameters: ", input_shape)
#     model = create_model(input_shape)
#     model_info = model.get_config()
#     model_architecture = model.summary()
#     print(model_architecture)
#     logging.info(f"The model architecture is as follows:")
#     model.summary(print_fn=logging.info)
# 
#    #Record start time to monitor training time
#     start = datetime.now()
#     logging.info(f"Training start time : {start}")    
# 
#     training_generator = DataGenerator(partition["train"],#X
#                                        label_dict,#y
#                                        dim=MAP_DIM,
#                                        batch_size=batch_size,
#                                        n_classes=4,
#                                        shuffle=True)
# 
# 
#     testing_generator = DataGenerator(partition["validate"],
#                                       label_dict,
#                                       dim=MAP_DIM,
#                                       batch_size=batch_size,
#                                       n_classes=4,
#                                       shuffle=False)#was True
# 
# 
#    #TO DO: need to find a way to run k-fold cross-validation during training
# 
# 
#     history = model.fit(
#         training_generator,
#         steps_per_epoch=int((len(X_train) / batch_size)),#len(X) if not using train-test-split
#         epochs=epochs,
#         validation_data=testing_generator,
#         validation_steps=(len(X_test) / batch_size),
#         use_multiprocessing=True,
#         workers=8)#8)
# 
# 
#    # Send history to csv
#     history_to_csv(history, histories_path / f"history.csv")
#     figure_from_csv(os.path.join(histories_path, "history.csv"),
#                     histories_path / f"history.png")
#    # Save model as h5
#    model.save(str(models_path / f"model.h5"))
#
#
#    #Record start time to monitor training time
#     end = datetime.now()
#     logging.info(f"Training end time : {end}")    
#     elapsed = end-start
#     logging.info(f"Training duration : {elapsed}")
#     print("Training duration: ", elapsed)
# 
#    # Make evaluation folder
# 
#     logging.info("Performing evaluation of model")
#     evaluation_dir_path = str(evaluations_path / f"evaluation")
#     if not Path(evaluation_dir_path).exists():
#       os.mkdir(evaluation_dir_path)
# 
#     logging.info("Getting predictions")
#     
##    print(int(math.ceil(len(X_test) / batch_size)))
##    print(int(np.round(len(X_test) / batch_size)))
# 
#     try:
#       predictions = model.predict(
#                           testing_generator,
#                           steps=math.ceil(len(X_test) / batch_size),
#                           verbose=1)
#       
# 
#       preds_rounded = np.round(predictions, 0)
#      #print("Predictions after rounding")
#      #print(preds_rounded)
#       
#       y_pred = np.argmax(preds_rounded, axis=1)
#       y_pred1 = preds_rounded.argmax(1)
# 
#       print("predicted labels ", y_pred)
#       print("known labels ", y_test[:-2])
#      #print(y_pred1)
#
#      #print("Length of predictions rounded: ", len(preds_rounded))
#     except Exception:
#       logging.warning("Could not round predictions")
#       raise
# 
#    #interim fix to be able to develop further; remove the last two samples in y_test
#    #y_test = y_test[:-2]
#    #print("Content of y_test")
#    #print(y_test[:-2])
#     try:
#       classes = ["class 1", "class 2", "class 3", "class 4", "class 5"]
#       labels = np.arange(5)
#       classification_metrics = metrics.classification_report(y_test[:-2], y_pred, labels=labels, target_names=classes)
#       print(classification_metrics)
#       logging.info(f"Multi-class classification report")
#       logging.info(classification_metrics)
#     except Exception:
#       logging.warning("Could not get multi-class classification report")
#       raise
# 
#     logging.info("Drawing confusion matrix.")
#     try:
##      cat_labels = pd.DataFrame(y_test[:-2].idxmax(axis=1))
##      cat_preds = pd.DataFrame(y_pred.idxmax(axis=1))
#       cat_labels = pd.DataFrame(y_test[:-2])
#       cat_preds = pd.DataFrame(y_pred)
# 
#       confusion_matrix = metrics.confusion_matrix(cat_labels, cat_preds)
#     except Exception:
#       logging.warning("Could not calculate confusion matrix")
#       raise
##    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
##      #Add Normalization Option
##      if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##        print("Normalized confusion matrix")
##      else:
##        print("Confusion matrix withou normalization")
##      #print(cm)
#       
#       
#     try:  
#       def draw_conf_mat(matrix):
#         datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
#         labels = ['0', '1']
#         ax = plt.subplot()
#         sns.heatmap(matrix, annot=True, ax=ax)
#         plt.title('Confusion matrix')
#         ax.set_xticklabels(labels)
#         ax.set_yticklabels(labels)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.savefig(str(evaluations_path / f"confusion_matrix_{datestring}.png"))
#         plt.close()
# 
#       draw_conf_mat(confusion_matrix)
# 
#     except Exception:
#       logging.warning("Could not draw confusion matrix")
#       raise
#       
# 
#       
#       
#       
#       
#       
#       
#       
#       
#     logging.info("Evaluations complete.")
# 
# 
#    # Load the model config information as a yaml file
#     with open(output_dir_path / "model_info.yaml", "w") as f:
#         yaml.dump(model_info, f)
# 
#    # Try to copy log file if it was created in training.log
#     try:
#         shutil.copy("training.log", output_dir_path)
#     except FileExistsError:
#         logging.warning("Could not find training.log to copy")
#     except Exception:
#         logging.warning("Could not copy training.log to output directory")


def pipeline_from_command_line(
    create_model: Callable[[int, int, int], Model], rgb: bool = False #2D
):
    """
    Run the training pipeline from the command line with config file

    Get parameters from the command line and pass them to training_pipeline in the parameter dict

    :param create_model: function which returns new Keras model to train and evaluate
    :param rgb: whether the model is expecting a 3 channel image
    """

    # Get from pipeline
    argument_dict = get_pipeline_parameters()
    
    print("received arguments: ", argument_dict)


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
        "--image_dim",
        required=True,
        type=int,
        nargs=2,
        help="X and Y dimensions of the map image slices"
    )
    parser.add_argument(
        "--runs",
        required=True,
        type=int,
        help="number of folds to run training and validation on. Must be equal or lower than k_folds",
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
    parser.add_argument(
        "--test_dir",
        help="directory with images for testing and producing classification reports. Leave empty if you do not want to perform evaluation",
    )
    parser.add_argument(
        "--slices_per_structure",
        type=int,
        help="number of images for each structure. To be used in testing only",
    )

    (known_args, unknown_args) = parser.parse_known_args()

#    assert known_args.k_folds >= known_args.runs, (
#        f"Number of runs must be less than or equal to k_folds, "
#        f"got {known_args.runs} runs and {known_args.k_folds} folds"
#    )
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
