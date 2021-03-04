"""Load a model and evaluate its performance against an unknown test set"""
import glob
import logging
import os
import re
import sqlite3
from pathlib import Path

import configargparse
import keras.models
import numpy as np
import pandas
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from modules.cnn.training_models.data_generator_resnet import DataGenerator

#MAP_DIM = (201, 201, 201)
MAP_DIM = (101, 101, 101)
#MAP_DIM = (51, 51, 51)

def evaluate(
    model_file: str,
#    test_dir: str,
#    database_file: str,
    test_set,
    output_dir: str,
    rgb: bool = False,
):
    # Load model
    try:
        model = keras.models.load_model(model_file)
    except Exception:
        logging.error(f"Failed to load model from {model_file}")
        raise

    logging.info(f"Model loaded from {model_file}")

#    # Get test files prepared
#    # Load files
#    try:
#        test_files = glob.glob(f"{test_dir}/*")
#        logging.info(f"Found {len(test_files)} files for testing")
#        assert len(test_files) > 0, f"Could not find files at {test_dir}"
#    except AssertionError as e:
#        logging.error(e)
#        raise
#    except Exception as e:
#        logging.error(e)
#        raise
#
#    try:
#        conn = sqlite3.connect(database_file)
#    except Exception:
#        logging.error(f"Could not connect to database at {database_file}")
#        raise
#
#    # Read table into pandas dataframe
#    data = pandas.read_sql(f"SELECT * FROM ai_labels", conn)
#    data_indexed = data.set_index("Name")
#
#    names = [re.findall("(.*)", Path(file).stem)[0] for file in test_files]
#    test_labels = [data_indexed.at[name, "Label"] for name in names]

    # Create training dataframe
    testing_dataframe = pandas.DataFrame({"Files": test_files, "Labels": test_labels})
    
    testing_dict = testing_dataframe.to_dict()

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_batch_size = len(testing_dataframe)

    # Add color mode selection, this is necessary for testing pretrained models which are expecting RGB images
    if rgb:
        color_mode = "rgb"
    else:
        color_mode = "grayscale"

    testing_generator = DataGenerator(testing_dict['Files'],
                                       testing_dict['Labels'],
                                       dim=MAP_DIM,
                                       batch_size=test_batch_size,
                                       n_classes=2,
                                       shuffle=True)

    logging.info("Getting predictions")

    try:
      predictions = model.predict_generator(
                          testing_generator,
                          steps=int(len(testing_dataframe["Files"]) / test_batch_size)
                          )
    except ValueError:
      logging.exception(
              "Ensure the RGB option is set correctly for your model - "
              "Some models expect 3 channel data")
      raise          

    # Per map analysis
    predictions_1 = [x for x in predictions if x[1] > x[0]]
    predictions_0 = [x for x in predictions if x[1] < x[0]]
    logging.info(f"Predicted good value {len(predictions_1)} times")
    logging.info(f"Predicted bad value {len(predictions_0)} times")

    predictions_decoded = [int(pred[1] > pred[0]) for pred in predictions]

    # Create an output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        # Make one
        try:
            os.mkdir(output_dir_path)
            logging.info(f"Created new directory in {output_dir_path}")
        except Exception as e:
            logging.error(
                f"Could not create directory at {output_dir_path}.\n"
                f"Please check permissions and location."
            )
            logging.error(e)
            raise

    # Save raw predictions
    raw_dataframe = pandas.DataFrame(
        {
            "File": testing_dataframe["Files"],
            "0": predictions[:, 0],
            "1": predictions[:, 1],
            "True Score": test_labels,
        }
    )
    raw_dataframe.set_index("File", inplace=True)
    raw_dataframe.to_csv(output_dir_path / "raw_predictions.csv")

    logging.info("Per map analysis:")
    per_map_class = classification_report(
        predictions_decoded, testing_dataframe["Labels"], output_dict=True
    )
    per_map_class_frame = pandas.DataFrame(per_map_class).transpose()
    per_map_conf = confusion_matrix(predictions_decoded, testing_dataframe["Labels"])
    per_map_conff_frame = pandas.DataFrame(per_map_conf)
    logging.info(per_map_class)
    logging.info(per_map_conf)
    # Save reports to csv
    per_map_class_frame.to_csv(output_dir_path / "per_map_class.csv")
    per_map_conff_frame.to_csv(output_dir_path / "per_map_conf.csv")
    logging.info("Per map analysis complete")

    logging.info("Evaluations complete.")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="Perform and record evaluation of a model from test files using per image predictions, "
        "predictions based on the average score of all the slices in a structure, "
        "and predictions based on counting predictions. "
        "Results given as raw scores per prediction, classification report and confusion matrix.",
    )

    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="yaml config file to provide parameters in",
    )
    parser.add_argument(
        "--model_file", required=True, help=".h5 file to load the model from"
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="directory of images to use as test data - should not form part of model training or validation data",
    )
    parser.add_argument(
        "--database_file",
        required=True,
        help="sqlite3 database with labels for test images",
    )

    parser.add_argument(
        "--output_dir", required=True, help="directory to output results files to"
    )

    parser.add_argument(
        "--rgb",
        action="store_true",
        default=False,
        help="set this parameter if the model is expecting rgb images",
    )

    args = parser.parse_args()

    evaluate(
        args.model_file,
        args.test_dir,
        args.database_file,
        args.output_dir,
        args.rgb,
    )
