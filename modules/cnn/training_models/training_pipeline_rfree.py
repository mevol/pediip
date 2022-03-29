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
import os
import shutil
import configargparse
import yaml
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow

from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from datetime import datetime
from typing import Callable
from pathlib import Path
from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
from modules.cnn.training_models.k_fold_boundaries import k_fold_boundaries
#from modules.cnn.evaluate_model import evaluate
from modules.cnn.training_models.data_generator_binary_rfree import DataGenerator
from modules.cnn.prepare_training_data_random_pick_combined import prepare_training_data_random_pick_combined

print("TensorFlow version: ", tensorflow.__version__)

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
        parameters_dict["output_dir"] + "_" + datetime.now().strftime("%Y%m%d_%H%M"))
    histories_path = output_dir_path / "histories"
    models_path = output_dir_path / "models"
    evaluations_path = output_dir_path / "evaluations"

    # Check if output directory exists
    if not output_dir_path.exists():
        # if it doesn't exist, make one
        try:
            # also make subdirectories for different outputs
            os.mkdir(output_dir_path)
            os.mkdir(histories_path)
            os.mkdir(models_path)
            os.mkdir(evaluations_path)
            logging.info(f"Created output directories at {output_dir_path} \n")
        except Exception:
            logging.exception(
                f"Could not create directory at {output_dir_path}.\n"
                f"Please check permissions and location. \n")
            raise

    # Log parameters
    logging.info(f"Running with parameters: {parameters_dict} \n")

    # Log the key information about the model and run
    with open(output_dir_path / "parameters.yaml", "w") as f:
        yaml.dump(parameters_dict, f)

    STACK_DIM = tuple((parameters_dict["slices_per_structure"],
                      parameters_dict["image_dim"][0],
                      parameters_dict["image_dim"][1]))

    # Check if input CSV holding sample filepaths does exist and open the file
    try:
        training_dir_path = Path(parameters_dict["sample_lable_lst"])
        assert (training_dir_path.exists())
        # Load data CSV file with filenames and labels
        data = pd.read_csv(training_dir_path)
        logging.info(f"Found {len(data)} samples for training \n")
    except Exception:
        logging.error(f"Could not open input map list \n")

    # separate data X from labels y; assigning labels based on rfree
    X = data[['filename', 'protocol', 'stage']]
    condition = (data['rfree'] < 0.5)
    data['ai_lable'] = np.where(condition, 1, 0)
    y = data['ai_lable']

    # getting the class distribution
    class_frequency = data.groupby(y).size()
    logging.info(f"Number of samples per class {class_frequency} \n")

    
    # split the data into training and test set; this is splitting the input CSV data;
    # and an additional challenge set of 5% of the data; this latter set is used to
    # finally challenge the algorithm after training and testing
    # create a 5% challenge set if needed
    X_temp, X_challenge, y_temp, y_challenge = train_test_split(X, y, test_size=0.05,
                                                                random_state=42)

    logging.info(f"Length of challenge data: {len(X_challenge)} \n")

    # use the remaining data for 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2,
                                                        random_state=100)
    logging.info(f"Number of samples in y_test: {len(y_test)} \n")
    logging.info(f"Number of samples in X_test: {len(X_test)} \n")
    logging.info(f"Number of samples in y_train: {len(y_train)} \n")
    logging.info(f"Number of samples in X_train: {len(X_train)} \n")

    print(X_test)
    print(00000000, len(X_test))

#    partition = {"train" : X_train,
#                 "validate" : X_test}
#    logging.info(f"Length of partition train: {len(partition['train'])} \n")
#    logging.info(f"Length of partition validate: {len(partition['validate'])} \n")
    
    # get the number of samples that need to be created to fill a batch for prediction
    num_batches_test = int(np.round(len(X_test) / parameters_dict["batch_size"]))
    print(1111111111, num_batches_test)
    num_batches_test_needed = int(math.ceil(len(X_test) / parameters_dict["batch_size"]))
    print(2222222222, num_batches_test_needed)
    batches_times_rounded_down = parameters_dict["batch_size"] * num_batches_test_needed
    print(3333333333, batches_times_rounded_down)
    diff_batch_samples = int( batches_times_rounded_down - len(X_test))
    print(555555555555, diff_batch_samples)

    # creating a dictionary for the label column to match sample ID with label
    label_dict = y.to_dict()
    last_y_key = list(label_dict.keys())[-1]
    print("Last y key: ", last_y_key)
    new_keys = last_y_key + diff_batch_samples
    last_y = y_test.iloc[-1]
    last_X = X_test.iloc[-1].values
    
    print(X_test)
    print(X_test.index)

    for i in range(last_y_key + 1, new_keys + 1):
        print(i)
        label_dict[i] = last_y
        y_test.loc[i] = last_y
        print(last_X)
        X_test.loc[i] = last_X
#        np.append(X_test, last_X, axis=0)
#        rep = 2
#        last = np.repeat(last_X,repeats= rep-1 ,axis=0)

#        X_test = np.vstack([X_test, last_X])

    print(X_test)

    partition = {"train" : X_train,
                 "validate" : X_test}
    logging.info(f"Length of partition train: {len(partition['train'])} \n")
    logging.info(f"Length of partition extended validate: {len(partition['validate'])} \n")
    print(len(label_dict))
    print(len(partition['validate']))


    assert len(label_dict) == len(partition['validate']) + len(partition['train']) + len(X_challenge)

#    additional_samples = pd.DataFrame(np.repeat(last_X, diff_batch_samples, axis=0))#last.values
#    print(additional_samples)
#    extend_X_test = pd.concat([X_test, additional_samples], ignore_index=True)
#    print(extend_X_test)
#    print("Index of last 20 rows: ", extend_X_test.iloc[-20:])
#    print("Index of last 15 rows: ", extend_X_test.iloc[- diff_batch_samples:])


#    assert len(partition['validate']) == len(X_test) + len(additional_samples)

    
#    print(last_y)
#    additional_y = pd.DataFrame(np.repeat(last_y, diff_batch_samples, axis=0))#last.values
#    print(len(additional_y))
#    additional_y_dict = additional_y.to_dict()
#    print(len(additional_y_dict))
#    print(additional_y_dict)
#    label_dict = label_dict.update(additional_y_dict)
    
    # set input dimensions for images and number of channels based on whether color or
    # grayscale is used
    if parameters_dict["rgb"] is True:
        logging.info("Using 3 channel image input to model")
        input_shape = (parameters_dict["slices_per_structure"],
                       parameters_dict["image_dim"][0],
                       parameters_dict["image_dim"][1],
                       3) #2D
        color_mode = "rgb"
    else:
        logging.info("Using single channel image input to model \n")
        input_shape = (parameters_dict["slices_per_structure"],
                       parameters_dict["image_dim"][0],
                       parameters_dict["image_dim"][1],
                       1) #2D
        color_mode = "grayscale"

    # Prepare data generators to get data out
    # Build model
    # Model run parameters
    epochs = parameters_dict["epochs"]
    batch_size = parameters_dict["batch_size"]

    # New model
    model = create_model(input_shape)
    model_info = model.get_config()
    model_architecture = model.summary()
    logging.info(f"The model architecture is as follows: \n")
    model.summary(print_fn=logging.info)

    #Record start time to monitor training time
    start = datetime.now()
    logging.info(f"Training start time : {start} \n")

    training_generator = DataGenerator(
                                       parameters_dict["xyz_limits"],
                                       parameters_dict["slices_per_axis"],
                                       partition["train"],#X
                                       label_dict,#y_train,#y
                                       dim=STACK_DIM,
                                       batch_size=batch_size,
                                       n_classes=2,
                                       shuffle=True)

    testing_generator = DataGenerator(
                                      parameters_dict["xyz_limits"],
                                      parameters_dict["slices_per_axis"],
                                      partition["validate"],
                                      label_dict,#y_test,
                                      dim=STACK_DIM,
                                      batch_size=batch_size,
                                      n_classes=2,
                                      shuffle=False)#was True

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

    #Record end time to monitor training time
    end = datetime.now()
    logging.info(f"Training end time : {end} \n")
    elapsed = end-start
    logging.info(f"Training duration : {elapsed} \n")
    print("Training duration: ", elapsed)

    # getting predictions on the testing data
    logging.info("Getting predictions \n")

    predict_steps = int(math.ceil(len(X_test) / batch_size))
    print("Steps to run until prediction finished: ", predict_steps)

    try:
      predictions = model.predict(
                          testing_generator,
                          steps=predict_steps,#predict_steps
                          verbose=1)

#      preds_rounded = np.round(predictions, 0)
#      #print("Predictions after rounding")
#      #print(preds_rounded)

#      y_pred = np.argmax(preds_rounded, axis=1)
#      y_pred1 = preds_rounded.argmax(1)

      print("predicted labels for the test set ", predictions)
      print("Length of predictions: ", len(predictions))
      print("known labels for the test set ", y_test)
      print("Length of y_test: ", len(y_test))
      #print(y_pred1)

      #print("Length of predictions rounded: ", len(preds_rounded))
    except Exception:
      logging.warning("Could not round predictions \n")
      raise

    #interim fix to be able to develop further; remove the last two samples in y_test
    #y_test = y_test[:-2]
    #print("Content of y_test")
    #print(y_test[:-2])
    try:
      classes = ["class 0", "class 1"]
      labels = np.arange(2)
      classification_metrics = metrics.classification_report(y_test, predictions,
                                                      labels=labels, target_names=classes)
      print(classification_metrics)
      logging.info(f"Classification report")
      logging.info(classification_metrics)
    except Exception:
      logging.warning("Could not get classification report \n")
      raise

    logging.info("Drawing confusion matrix. \n")
    try:
#      cat_labels = pd.DataFrame(y_test[:-2].idxmax(axis=1))
#      cat_preds = pd.DataFrame(y_pred.idxmax(axis=1))
      cat_labels = pd.DataFrame(y_test)
      cat_preds = pd.DataFrame(predictions)

      confusion_matrix = metrics.confusion_matrix(cat_labels, cat_preds)
    except Exception:
      logging.warning("Could not calculate confusion matrix \n")
      raise

    try:  
      def draw_conf_mat(matrix):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        labels = ['0', '1']
        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax=ax)
        plt.title('Confusion matrix')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(str(evaluations_path / f"confusion_matrix_{datestring}.png"))
        plt.close()

      draw_conf_mat(confusion_matrix)

    except Exception:
      logging.warning("Could not draw confusion matrix. \n")
      raise
      
      
    # Make evaluation folder to use the challenge data
    logging.info("Performing evaluation of model \n")
    evaluation_dir_path = str(evaluations_path / f"evaluation")
    if not Path(evaluation_dir_path).exists():
      os.mkdir(evaluation_dir_path)
      evaluate(
                str(models_path / f"model.h5"),
                parameters_dict["test_dir"],
                evaluation_dir_path,
                parameters_dict["sample_lable_lst"],
                parameters_dict["slices_per_structure"],
                rgb=parameters_dict["rgb"],
            )
    else:
            logging.info(
              f"Requires test directory and slices_per_structure for evaluation. \n"
              f"No evaluation performed \n"
            )

    # Load the model config information as a yaml file
    with open(output_dir_path / "model_info.yaml", "w") as f:
        yaml.dump(model_info, f)

    # Try to copy log file if it was created in training.log
    try:
        shutil.copy("training.log", output_dir_path)
    except FileExistsError:
        logging.warning("Could not find training.log to copy \n")
    except Exception:
        logging.warning("Could not copy training.log to output directory \n")


def pipeline_from_command_line(
    create_model: Callable[[int, int, int, int], Model], rgb: bool = False):
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
    ), f"Must provide bool for rgb, got {type(rgb)} of value {rgb} \n"
    argument_dict["rgb"] = rgb

    logging.info(f"Running model with parameters: {argument_dict} \n")

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
        " The images will automatically be rescaled.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        is_config_file=True,
        help="config file to specify the following parameters in")
    parser.add_argument(
        "--sample_lable_lst",
        required=True,
        help="list of sample files with associated training lables")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory to output results files to. Will be appended with date and time of program run")
    parser.add_argument(
        "--xyz_limits",
        type=int,
        nargs=3,
        required=True,
        help="xyz size of the output map file")
    parser.add_argument(
        "--image_dim",
        required=True,
        type=int,
        nargs=2,
        help="X and Y dimensions of the map image slices")
    parser.add_argument(
        "--k_folds",
        required=True,
        type=int,
        help="number of folds to create for k-fold cross-validation")
    parser.add_argument(
        "--runs",
        required=True,
        type=int,
        help="number of folds to run training and validation on. Must be equal or lower than k_folds")
    parser.add_argument(
        "--epochs", type=int,
        required=True,
        help="number of epochs to use in each run")
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        help="size of batch to load during training and validation. Should be exact factor of the number of images provided")
    parser.add_argument(
        "--slices_per_axis",
        required=True,
        type=int,
        help="number of slices to be produced for each axis of the standard volume")
    parser.add_argument(
        "--slices_per_structure",
        required=True,
        type=int,
        help="number of images for each structure. To be used in testing only")

    (known_args, unknown_args) = parser.parse_known_args()

    assert known_args.k_folds >= known_args.runs, (
        f"Number of runs must be less than or equal to k_folds, \n"
        f"got {known_args.runs} runs and {known_args.k_folds} folds. \n")
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
            f"Could not find image_augmentation_dict in {known_args.config}, \n"
            f"performing scaling only. \n")
        image_augmentation_dict = {}
    assert isinstance(
        image_augmentation_dict, dict
    ), f"image_augmentation_dict must be provided as a dictionary in YAML, got {image_augmentation_dict} \n"

    argument_dict["image_augmentation_dict"] = image_augmentation_dict


    return argument_dict


if __name__ == "__main__":

    args = get_pipeline_parameters()
    print(f"Received the following arguments: {args} \n")

