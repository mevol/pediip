import argparse
import glob
import logging
import os
from pathlib import Path


def list_temp_files(directory):
    """List all temporary files within a directory which are marked with *temp* in their name"""
    try:
        dir_path = Path(directory)
        assert dir_path.exists()
    except Exception:
        raise Exception(f"Expected absolute path to valid directory, got {directory}")

    temps = glob.glob(str(dir_path / "*temp*"))

    temp_files = [file for file in temps if Path(file).is_file()]

    return temp_files


def delete_file(filename):
    """Delete the file and return True"""
    try:
        file_path = Path(filename)
        assert file_path.exists(), f"Could not find file to delete at {file_path}"
        os.remove(file_path)
    except Exception:
        logging.error(f"Could not delete file at {filename}")
        raise

    return True


def delete_temp_files(directory):
    """Delete all temporary files in the directory and return True when complete"""
    logging.debug(f"Deleting all files in {directory}")
    try:
        temp_files = list_temp_files(directory)
    except Exception:
        raise

    try:
        for file in temp_files:
            delete_file(file)
    except Exception:
        raise

    return True


if __name__ == "__main__":
    # As command line utility, check user wants to do this
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        help="directory provided is absolute path, otherwise assumed relative",
        action="store_true",
    )
    parser.add_argument(
        "directory",
        help="the directory you wish to remove temporary files from",
        type=str,
    )
    parser.add_argument(
        "--force", help="delete all temp files without checking", action="store_true"
    )

    args = parser.parse_args()

    if args.a == True:
        dir_name = args.directory
    else:
        dir_name = Path(os.getcwd()) / args.directory

    if args.force == True:
        delete_temp_files(dir_name)
        print("All temp files deleted")
    else:
        temp_files = list_temp_files(dir_name)
        print("Found following temp files:")
        for file in temp_files:
            print(Path(file).name)
        delete = input("Are you sure you want to delete all of these files? [y/N]")
        if delete == "y":
            delete_temp_files(dir_name)
            print("All temp files deleted")
