"""
Pipeline for training models with cross validation, recording parameters and performing evaluation.

Designed to make it easier to create and evaluate models with different architectures with the
same training parameters.
"""

# Necessary to make the run as consistent as possible
import logging
import os
import shutil
import tensorflow
import pandas as pd
import configargparse
import pandas
import yaml
import numpy as np
import math

from pathlib import Path
from datetime import datetime
from typing import Callable
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from modules.cnn.training_models.plot_history import confusion_matrix_and_stats
from modules.cnn.training_models.plot_history import plot_precision_recall_vs_threshold
from modules.cnn.training_models.plot_history import plot_roc_curve, confusion_matrix_and_stats_multiclass
from modules.cnn.training_models.plot_history import reg_history_to_csv, reg_figure_from_csv
from modules.cnn.training_models.k_fold_boundaries import k_fold_boundaries
from modules.cnn.training_models.data_generator_volume_regression import DataGenerator

print(tensorflow.__version__)

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

  if not output_dir_path.exists():
    # Make one
    try:
      # Make directories
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

  MAP_DIM = tuple((parameters_dict["xyz_limits"][0],
                   parameters_dict["xyz_limits"][1],
                   parameters_dict["xyz_limits"][2]))

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
  X = data['filename']

  y = data['phase_error']#regression

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
  logging.info(f"Number of samples in y_test: {len(y_test)} \n"
               f"Number of samples in X_test: {len(X_test)} \n"
               f"Number of samples in y_train: {len(y_train)} \n"
               f"Number of samples in X_train: {len(X_train)} \n"
               f"Number of samples in y_challenge: {len(y_challenge)} \n"
               f"Number of samples in X_challenge: {len(X_challenge)} \n")

  # get the number of samples that need to be created to fill a batch for prediction
  # y_test and X_test
  num_batches_test_needed = int(math.ceil(len(X_test) / parameters_dict["batch_size"]))
  batches_times_rounded_down = parameters_dict["batch_size"] * num_batches_test_needed
  diff_batch_samples = int( batches_times_rounded_down - len(X_test))

  # creating a dictionary for the label column to match sample ID with label
  label_dict = y.to_dict()
  # checking y length to extend missing y_test
  last_y_key = list(label_dict.keys())[-1]
  # get the ID of the last sample to expand from there
  new_keys = last_y_key + diff_batch_samples
  # getting last sample of y_test and X_test
  last_y = y_test.iloc[-1]
  last_X = X_test.iloc[-1]

  for i in range(last_y_key + 1, new_keys + 1):
    label_dict[i] = last_y
    y_test.loc[i] = last_y
    X_test.loc[i] = last_X

  # get the number of samples that need to be created to fill a batch for prediction
  # y_challenge and y_test
  num_batches_challenge_needed = int(math.ceil(len(X_challenge) / parameters_dict["batch_size"]))
  batches_times_rounded_down2 = parameters_dict["batch_size"] * num_batches_challenge_needed
  diff_batch_samples2 = int( batches_times_rounded_down2 - len(X_challenge))

  # checking y length to extend missing y_test
  last_y_key2 = list(label_dict.keys())[-1]
  # get the ID of the last sample to expand from there
  new_keys2 = last_y_key2 + diff_batch_samples2
  # getting last sample of y_test and X_test
  last_challenge_X = X_challenge.iloc[-1]
  last_challenge_y = y_challenge.iloc[-1]

  for i in range(last_y_key2 + 1, new_keys2 + 1):
    label_dict[i] = last_challenge_y
    y_challenge.loc[i] = last_challenge_y
    X_challenge.loc[i] = last_challenge_X
  
  
#  # repeat each train sample and lable multiple times to increase number of samples
#  train_merge = X_train.merge(y_train, left_index=True, right_index=True)
#  print(train_merge.columns)
#  print(train_merge.head())
#  
#  train_new = []
#  train_new.append([train_merge]*5,ignore_index=True)
#  
#  print(train_new)


  partition = {"train" : X_train,
               "validate" : X_test,
               "challenge" : X_challenge}
  logging.info(f"Length of partition train: {len(partition['train'])} \n"
               f"Length of partition extended validate: {len(partition['validate'])} \n"
               f"Length of partition extended challenge: {len(partition['challenge'])} \n")

  assert len(label_dict) == (len(partition['validate'])
                             + len(partition['train'])
                             + len(partition['challenge']))

  # set input dimensions for images and number of channels based on whether color or
  # grayscale is used
  if parameters_dict["rgb"] is True:
    logging.info("Using 3 channel image input to model")
    input_shape = (MAP_DIM[0], MAP_DIM[1], MAP_DIM[2], 3)
    color_mode = "rgb"
  else:
    logging.info("Using single channel image input to model \n")
    input_shape = (MAP_DIM[0], MAP_DIM[1], MAP_DIM[2], 1)
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
                                       partition["train"],#X
                                       label_dict,#y_train,#y
                                       dim=MAP_DIM,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       augmentation=True)

  testing_generator = DataGenerator(
                                      parameters_dict["xyz_limits"],
                                      partition["validate"],
                                      label_dict,#y_test,
                                      dim=MAP_DIM,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      augmentation=False)

  challenge_generator = DataGenerator(
                                      parameters_dict["xyz_limits"],
                                      partition["challenge"],
                                      label_dict,#y_test,
                                      dim=MAP_DIM,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      augmentation=False)

  history = model.fit(
        training_generator,
        steps_per_epoch=int((len(X_train) / batch_size)),#len(X) if not using train-test-split
        epochs=epochs,
        validation_data=testing_generator,
        validation_steps=(len(X_test) / batch_size),
        use_multiprocessing=True,
        workers=8)#8)

  # Send history to csv for regression
  reg_history_to_csv(history, histories_path / f"history.csv")
  reg_figure_from_csv(os.path.join(histories_path, "history.csv"),
                  histories_path / f"history.png")

  # Save model as h5
  model.save(str(models_path / f"model.h5"))

  #Record end time to monitor training time
  end = datetime.now()
  logging.info(f"Training end time : {end} \n")
  elapsed = end-start
  logging.info(f"Training duration : {elapsed} \n")
  print("Training duration: ", elapsed)

  # getting predictions on the testing data to evaluate the model
  # Make evaluation folder to use the challenge data
  logging.info("Performing evaluation of model using X_test and X_challenge \n")
  date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
  # calculate the number of steps to be used in prediction and model evaluation
  # X_test
  predict_steps = int(math.ceil(len(X_test) / batch_size))
  logging.info(f"Steps to run until prediction for X_test finished: {predict_steps} \n")
  try:
    logging.info("Getting predictions for X_test\n")
    y_pred = model.predict(
                          testing_generator,
                          steps=predict_steps,
                          verbose=1)
    # rounding the prediction probabilities to integers for 0/1 classes in binary case
    preds_rounded = np.round(y_pred, 0)
    logging.info(f"Turning probabilities for X_test into integers \n")
    # only get the class1 results
    y_pred1 = np.argmax(preds_rounded, axis=1)
  except Exception:
    logging.warning("Could not round predictions for X_test\n")
    raise

  # calculate the number of steps to be used in prediction and model evaluation
  # X_challenge
  challenge_steps = int(math.ceil(len(X_challenge) / batch_size))
  logging.info(f"Steps to run until prediction finished: {challenge_steps} \n")
  try:
    logging.info("Getting predictions for challenge data \n")
    y_pred_challenge = model.predict(
                          challenge_generator,
                          steps=challenge_steps,
                          verbose=1)
    # rounding the prediction probabilities to integers for 0/1 classes in binary case
    preds_challenge_rounded = np.round(y_pred_challenge, 0)
    logging.info(f"Turning probabilities into into integers \n")
    # only get the class1 results
    y_pred_challenge1 = np.argmax(preds_challenge_rounded, axis=1)
  except Exception:
    logging.warning("Could not round predictions \n")
    raise

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
  create_model: Callable[[int, int, int, int], Model], rgb: bool = False ):
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
      rgb, bool), f"Must provide bool for rgb, got {type(rgb)} of value {rgb}"
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
    " The images will automatically be rescaled.",)

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
      "--xyz_limits",
      type=int,
      nargs=3,
      required=True,
      help="xyz size of the output map file")
  parser.add_argument(
      "--output_dir",
      required=True,
      help="directory to output results files to. Will be appended with date and time of program run")
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

  args = parser.parse_args()

  assert args.k_folds >= args.runs, (
    f"Number of runs must be less than or equal to k_folds, \n"
    f"got {args.runs} runs and {args.k_folds} folds. \n")

  argument_dict = vars(args)

  return argument_dict


if __name__ == "__main__":

  args = get_pipeline_parameters()
  print(f"Received the following arguments: {args}")
