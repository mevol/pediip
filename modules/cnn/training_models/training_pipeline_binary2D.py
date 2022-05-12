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
#from typing import Tuple
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
from glob import glob
from itertools import chain

from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
from modules.cnn.training_models.k_fold_boundaries import k_fold_boundaries
from modules.cnn.evaluate_model import evaluate
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
      logging.exception(f"Could not create directory at {output_dir_path}.\n"
                        f"Please check permissions and location. \n")
      raise

  # Log parameters
  logging.info(f"Running with parameters: {parameters_dict} \n")

  # Log the key information about the model and run
  with open(output_dir_path / "parameters.yaml", "w") as f:
    yaml.dump(parameters_dict, f)

  IMG_DIM = tuple(parameters_dict["image_dim"])
  print("image dimensions ", IMG_DIM)

  # Load training files
  try:
    image_dir_path = Path(parameters_dict["training_dir"])
    assert (image_dir_path.exists()), f"Could not find directory at {image_dir_path} \n"
    train_files = [str(file) for file in image_dir_path.iterdir()]
    assert len(train_files) > 0, f"Found no files in {image_dir_path} \n"
    print(len(train_files))
    #print(train_files)
    logging.info(f"Found {len(train_files)} files for training \n")
  except Exception:
    logging.error(f"Found no files in {image_dir_path} \n")
    raise
  try:
    training_dir_path = Path(parameters_dict["sample_lable_lst"])
    assert (training_dir_path.exists())
    # Load data CSV file with filenames and labels
    data = pd.read_csv(training_dir_path)
    logging.info(f"Found {len(data)} samples for training \n")
  except Exception:
    logging.error(f"Could not open input map list \n")
    raise

  # creating test, train and validation set
  # separate data X from labels y; assigning labels based on rfree
  X = data[['filename', 'protocol', 'stage']]
  y = data['ai_label']

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
  last_X = X_test.iloc[-1].values

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
  last_challenge_X = X_challenge.iloc[-1].values
  last_challenge_y = y_challenge.iloc[-1]

  for i in range(last_y_key2 + 1, new_keys2 + 1):
    label_dict[i] = last_challenge_y
    y_challenge.loc[i] = last_challenge_y
    X_challenge.loc[i] = last_challenge_X

  partition = {"train" : X_train,
               "validate" : X_test,
               "challenge" : X_challenge}
  logging.info(f"Length of partition train: {len(partition['train'])} \n"
               f"Length of partition extended validate: {len(partition['validate'])} \n"
               f"Length of partition extended challenge: {len(partition['challenge'])} \n")

  assert len(label_dict) == (len(partition['validate'])
                             + len(partition['train'])
                             + len(partition['challenge']))

#  column_names = ['filename', 'protocol', 'stage', 'ai_label']
  train_concat = pd.concat([X_train, y_train], axis = 1, ignore_index = True)
  print(train_concat.columns)
  print(train_concat.head())
  train_concat.rename(columns = {'0':'filename', '1':'protocol',
                                  '2':'stage', '3':'ai_label'}, inplace = True)
  print(train_concat.columns)
  test_concat = pd.concat([X_test, y_test], axis = 1, ignore_index = True, names=X.columns)
  test_concat.columns = column_names
  challenge_concat = pd.concat([X_challenge, y_challenge], axis = 1, ignore_index = True, names=X.columns)
  challenge_concat.columns = column_names
  # expand X_train, X_test and X_validation from single MTZ to contain
  # the corresponding 60 image slices
  def get_sample_name(x):
    try:
      target_file = x.split("/")[-1]
      target_file_stripped = target_file.strip('.mtz')
    except Exception:
      pass
    try:
      target_name = x.split("/")[8]
    except Exception:
      pass
    try:
      homo = x.split("/")[12]
    except Exception:
      homo = "none"
      logging.error(f"Could not find homologue to work with. \n")
      pass
    name = target_name+"_"+homo+"_"+target_file_stripped
    return name
  
  train_concat['name'] = train_concat.loc[:, 'filename'].apply(get_sample_name)
  print(train_concat)
  print(len(train_concat))
  test_concat['name'] = test_concat.loc[:, 'filename'].apply(get_sample_name)
  print(test_concat)
  print(len(test_concat))
  challenge_concat['name'] = challenge_concat.loc[:, 'filename'].apply(get_sample_name)
  print(challenge_concat)
  print(len(challenge_concat))


#  def expand_sets(working_set, images):
##    temp = pd.concat([X_set, y_set], axis = 1, ignore_index = True)
#    temp_indexed = working_set.set_index('name')
#    df = pd.DataFrame(columns = ["filename", "ai_label"])
#    for i in range(len(temp)):
#      sample = temp.iloc[i, 0]
#      label = temp.iloc[i, -1]
#      sample_split = sample.split('/')
#      name = sample_split[-1]
#      target_name = sample_split[8]
#      homo = sample_split[12]
#      sample_stem = name.strip('.mtz')
#      file_stem = target_name+"_"+homo+"_"+sample_stem
#      file_list = (glob(os.path.join(image_dir_path, file_stem+"*.png")))
#      dummy = pd.DataFrame({'filename':file_list,
#                             'ai_label' : label})
#      df = pd.concat([df, dummy], axis=0, ignore_index=True)
#    return df
#
#  train_new = expand_sets(X_train, y_train, train_files)
#  test_new = expand_sets(X_test, y_test, train_files)
#  challenge_new = expand_sets(X_challenge, y_challenge, train_files)

  #Prepare data generators to get data out
  #Always rescale and also expand dictionary provided as parameter
  train_datagen = ImageDataGenerator(
         rescale=1.0 / 255, **parameters_dict["image_augmentation_dict"])
  #Only rescale test and validation
  test_datagen = ImageDataGenerator(rescale=1.0 / 255)
  validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
 
  #Build model
  if parameters_dict["rgb"] is True:
    logging.info("Using 3 channel image input to model")
    input_shape = (IMG_DIM[0], IMG_DIM[1], 3)
    color_mode = "rgb"
  else:
    logging.info("Using single channel image input to model")
    input_shape = (IMG_DIM[0], IMG_DIM[1], 1)
    color_mode = "grayscale"

#     #Create training dataframe
#     training_dataframe = pandas.DataFrame(
#         {"Files": train_files, "Labels": [str(label) for label in train_labels]}
#     )
#     print(training_dataframe.head())
#     training_dataframe.set_index("Files")
#     training_data_shuffled = training_dataframe.sample(frac=1)    
#         
#         
#     
#     #Train the model k-fold number of times on different folds and record the output
#     Model run parameters
#     k_folds = parameters_dict["k_folds"]
#     runs = parameters_dict["runs"]
#     epochs = parameters_dict["epochs"]
#     batch_size = parameters_dict["batch_size"]
# 
#     fold_boundaries = k_fold_boundaries(train_files, k_folds)
#     for k in range(runs):
#         logging.info(f"Running cross validation set {k + 1}")
# 
#         #New model
#         model = create_model(input_shape)
#         model_info = model.get_config()
# 
#         #Separate the active training and validations set based on the fold boundaries
#         active_training_set = pandas.concat(
#             [
#                 training_data_shuffled[: fold_boundaries[k][0]],
#                 training_data_shuffled[fold_boundaries[k][1] :],
#             ]
#         )
#         active_validation_set = training_data_shuffled[
#             fold_boundaries[k][0] : fold_boundaries[k][1]
#         ]
# 
#         logging.info(f"Active training set of {len(active_training_set['Files'])}")
#         logging.info(f"Active validation set of {len(active_validation_set['Files'])}")
# 
#         #Record start time to monitor training time
#         start = datetime.now()
#         logging.info(f"Training start time : {start}")    
# 
#         #Create generators
#         train_generator = train_datagen.flow_from_dataframe(
#             active_training_set,
#             x_col="Files",
#             y_col="Labels",
#             target_size=IMG_DIM,
#             color_mode=color_mode,
#             shuffle=True,
#             batch_size=batch_size,
#             class_mode="categorical",
#         )
# 
#         val_generator = validation_datagen.flow_from_dataframe(
#             active_validation_set,
#             x_col="Files",
#             y_col="Labels",
#             target_size=IMG_DIM,
#             color_mode=color_mode,
#             shuffle=True,
#             batch_size=batch_size,
#             class_mode="categorical",
#         )
# 
#         history = model.fit_generator(
#             train_generator,
#             steps_per_epoch=int((len(active_training_set["Files"]) / batch_size)),
#             epochs=epochs,
#             validation_data=val_generator,
#             validation_steps=(len(active_validation_set["Files"]) / batch_size),
#             use_multiprocessing=True,
#             workers=8,
#         )
# 
# 
#         #Send history to csv
#         history_to_csv(history, histories_path / f"history_{k}.csv")
#         figure_from_csv(histories_path / f"history_{k}.csv",
#                       histories_path / f"history_{k}.png")
#         #Save model as h5
#         model.save(str(models_path / f"model_{k}.h5"))
# 
# 
#         #Record end time to monitor training time
#         end = datetime.now()
#         logging.info(f"Training end time : {end}")    
#         elapsed = end-start
#         logging.info(f"Training duration : {elapsed}")
#         print("Training duration: ", elapsed)
# 
#         #Make evaluation folder to use the test data
# 
#        logging.info("Performing evaluation of model")
#        evaluation_dir_path = str(evaluations_path / f"evaluation")
#        if not Path(evaluation_dir_path).exists():
#          os.mkdir(evaluation_dir_path)
# 
# 
#         #Make evaluation folder
#         if parameters_dict["test_dir"] and parameters_dict["slices_per_structure"]:
#             logging.info("Performing evaluation of model")
#             evaluation_dir_path = str(evaluations_path / f"evaluation_{k}")
#             if not Path(evaluation_dir_path).exists():
#                 os.mkdir(evaluation_dir_path)
#             evaluate(
#                 str(models_path / f"model_{k}.h5"),
#                 parameters_dict["test_dir"],
#                 evaluation_dir_path,
#                 parameters_dict["sample_lable_lst"],
#                 parameters_dict["slices_per_structure"],
#                 rgb=parameters_dict["rgb"],
#             )
#         else:
#             logging.info(
#                 "Requires test directory and slices_per_structure for evaluation. "
#                 "No evaluation performed"
#             )
# 
#     #Load the model config information as a yaml file
#     with open(output_dir_path / "model_info.yaml", "w") as f:
#         yaml.dump(model_info, f)
# 
#     #Try to copy log file if it was created in training.log
#     try:
#         shutil.copy("training.log", output_dir_path)
#     except FileExistsError:
#         logging.warning("Could not find training.log to copy")
#     except Exception:
#         logging.warning("Could not copy training.log to output directory")
# 
# 
# 
# 
#         logging.info("Getting predictions")
# 
#        print(int(math.ceil(len(X_test) / batch_size)))
#        print(int(np.round(len(X_test) / batch_size)))
# 
# #TO DO: put evaluation function here and make a test generator to use the test data
# #       need to look at the old way by Tim to make sure things are correct
# #       code below should update "evaluate_model.py" and then "evaluate section" of
# #       code needs to be run
# 
# 
# 
# 
# 
# 
#         try:
#           predictions = model.predict(
#                            testing_generator,
#                            steps=math.ceil(len(X_test) / batch_size),
#                            verbose=1)
# 
#           preds_rounded = np.round(predictions, 0)
#          print("Predictions after rounding")
#          print(preds_rounded)
# 
#           y_pred = np.argmax(preds_rounded, axis=1)
#           y_pred1 = preds_rounded.argmax(1)
# 
#           print("predicted labels ", y_pred)
#           print("known labels ", y_test[:-2])
#          print(y_pred1)
# 
#          print("Length of predictions rounded: ", len(preds_rounded))
#         except Exception:
#           logging.warning("Could not round predictions")
#           raise


##########################################################################################

    #interim fix to be able to develop further; remove the last two samples in y_test
    #y_test = y_test[:-2]
    #print("Content of y_test")
    #print(y_test[:-2])
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
        "--k_folds",
        required=True,
        type=int,
        help="number of folds to create for k-fold cross-validation",
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

    assert known_args.k_folds >= known_args.runs, (
        f"Number of runs must be less than or equal to k_folds, "
        f"got {known_args.runs} runs and {known_args.k_folds} folds"
    )
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
