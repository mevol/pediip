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
import configargparse
import yaml
import math
import pandas as pd
import numpy as np

from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
from typing import Callable
from pathlib import Path
from modules.cnn.training_models.plot_history import history_to_csv, figure_from_csv
from modules.cnn.training_models.plot_history import confusion_matrix_and_stats
from modules.cnn.training_models.plot_history import plot_precision_recall_vs_threshold
from modules.cnn.training_models.plot_history import plot_roc_curve, confusion_matrix_and_stats_multiclass
from modules.cnn.training_models.k_fold_boundaries import k_fold_boundaries
from modules.cnn.training_models.data_generator_volume_mrc_classification import DataGenerator

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

    MAP_DIM = tuple((parameters_dict["map_dim"][0],
                     parameters_dict["map_dim"][1],
                     parameters_dict["map_dim"][2]))


    # Load training files
    try:
      maps_dir_path = Path(parameters_dict["maps_dir"])
      assert (training_dir_path.exists()), f"Could not find directory at {maps_dir} \n"
      all_files = [str(file) for file in maps_dir_path.iterdir()]
      assert len(all_files) > 0
      logging.info(f"Found {len(all_files)} files for training, testing and challenge \n")
    except Exception:
        logging.error(f"Found no files in {maps_dir} \n")

    try:
      training_dir_path = Path(parameters_dict["sample_lable_lst"])
      assert (training_dir_path.exists())
      # Load data CSV file with filenames and labels
      data = pd.read_csv(training_dir_path)
      logging.info(f"Found {len(data)} samples for training \n")
    except Exception:
      logging.error(f"Could not open input map list \n")

    # separate data X from labels y
    X = data['filename']
    y = data['ai_label']

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

    # set input dimensions for images and number of channels based on whether color or
    # grayscale is used
    if parameters_dict["rgb"] is True:
        logging.info("Using 3 channel image input to model")
        input_shape = (MAP_DIM, 3)
        color_mode = "rgb"
    else:
        logging.info("Using single channel image input to model \n")
        input_shape = (MAP_DIM, 1)
        color_mode = "grayscale"

    # Prepare data generators to get data out
    # Build model
    # Model run parameters
    epochs = parameters_dict["epochs"]
    batch_size = parameters_dict["batch_size"]
    num_classes = parameters_dict["num_classes"]

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
                                       partition["train"],#X
                                       label_dict,#y_train,#y
                                       dim=MAP_DIM,
                                       batch_size=batch_size,
                                       n_classes=num_classes,
                                       shuffle=True,
                                       augmentation=True)

    testing_generator = DataGenerator(
                                      partition["validate"],
                                      label_dict,#y_test,
                                      dim=MAP_DIM,
                                      batch_size=batch_size,
                                      n_classes=num_classes,
                                      shuffle=False,
                                      augmentation=False)

    challenge_generator = DataGenerator(
                                      partition["challenge"],
                                      label_dict,#y_test,
                                      dim=MAP_DIM,
                                      batch_size=batch_size,
                                      n_classes=num_classes,
                                      shuffle=False,
                                      augmentation=False)

    history = model.fit(
        training_generator,
        steps_per_epoch=int((len(X_train) / batch_size)),#len(X) if not using train-test-split
        epochs=epochs,
        validation_data=testing_generator,
        validation_steps=(len(X_test) / batch_size),
        use_multiprocessing=True,
        workers=8)


    # Send history to csv
    history_to_csv(history, histories_path / f"history.csv")
    figure_from_csv(os.path.join(histories_path, "history.csv"),
                    histories_path / f"history.png")
    # Save model as h5
    model.save(str(models_path / f"model.h5"))


    #Record start time to monitor training time
    end = datetime.now()
    logging.info(f"Training end time : {end}")    
    elapsed = end-start
    logging.info(f"Training duration : {elapsed}")
    print("Training duration: ", elapsed)

    # Make evaluation folder

    logging.info("Performing evaluation of model")
    evaluation_dir_path = str(evaluations_path / f"evaluation")
    if not Path(evaluation_dir_path).exists():
      os.mkdir(evaluation_dir_path)

    logging.info("Getting predictions")
    
#    print(int(math.ceil(len(X_test) / batch_size)))
#    print(int(np.round(len(X_test) / batch_size)))

    try:
      predictions = model.predict(
                          testing_generator,
                          steps=math.ceil(len(X_test) / batch_size),
                          verbose=1)
      

      preds_rounded = np.round(predictions, 0)
      #print("Predictions after rounding")
      #print(preds_rounded)
      
      y_pred = np.argmax(preds_rounded, axis=1)
      y_pred1 = preds_rounded.argmax(1)

      print("predicted labels ", y_pred)
      print("known labels ", y_test[:-2])
      #print(y_pred1)

      #print("Length of predictions rounded: ", len(preds_rounded))
    except Exception:
      logging.warning("Could not round predictions")
      raise

    #interim fix to be able to develop further; remove the last two samples in y_test
    #y_test = y_test[:-2]
    #print("Content of y_test")
    #print(y_test[:-2])
    try:
      classes = ["class 1", "class 2", "class 3", "class 4", "class 5"]
      labels = np.arange(5)
      classification_metrics = metrics.classification_report(y_test[:-2], y_pred, labels=labels, target_names=classes)
      print(classification_metrics)
      logging.info(f"Multi-class classification report")
      logging.info(classification_metrics)
    except Exception:
      logging.warning("Could not get multi-class classification report")
      raise

    logging.info("Drawing confusion matrix.")
    try:
#      cat_labels = pd.DataFrame(y_test[:-2].idxmax(axis=1))
#      cat_preds = pd.DataFrame(y_pred.idxmax(axis=1))
      cat_labels = pd.DataFrame(y_test[:-2])
      cat_preds = pd.DataFrame(y_pred)

      confusion_matrix = metrics.confusion_matrix(cat_labels, cat_preds)
    except Exception:
      logging.warning("Could not calculate confusion matrix")
      raise
#    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#      #Add Normalization Option
#      if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#      else:
#        print("Confusion matrix withou normalization")
#      #print(cm)
      
      
    try:  
      def draw_conf_mat(matrix):
        datestring = datetime.strftime(datetime.now(), '%Y%m%d_%H%M')
        labels = ['0', '1', '2', '3', '4']      
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
      logging.warning("Could not draw confusion matrix")
      raise
      

      
      
      
      
      
      
      
      
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
        " The images will automatically be rescaled.",)

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        is_config_file=True,
        help="config file to specify the following parameters in",)
    parser.add_argument(
        "--map_dir",
        required=True,
        help="directory with training images in")
    parser.add_argument(
        "--sample_lable_lst",
        required=True,
        help="list of sample files with associated training lables",)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="directory to output results files to. Will be appended with date and time of program run",)
    parser.add_argument(
        "--map_dim",
        type=int,
        nargs=3,
        required=True,
        help="dimensions of the input map")
    parser.add_argument(
        "--num_classes",
        required=True,
        type=int,
        help="number of classes")
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
        help="size of batch to load during training and validation. Should be exact factor of the number of images provided",)

    args = parser.parse_args()

    assert args.k_folds >= args.runs, (
        f"Number of runs must be less than or equal to k_folds, \n"
        f"got {args.runs} runs and {args.k_folds} folds. \n")

    argument_dict = vars(args)

    return argument_dict


if __name__ == "__main__":

    args = get_pipeline_parameters()
    print(f"Received the following arguments: {args}")
