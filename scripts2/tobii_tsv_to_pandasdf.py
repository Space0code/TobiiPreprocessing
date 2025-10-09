"""

If you need to get the raw tobii eye-tracker data from a tsv file, you only need to use get_subject_dfs(directory, tobii_dir_name, other_dir_name).
The other two are helper functions.

Usage example:

directory = r"C:\TrustMe\Eye Tracker\Datasets\playground\OCOsense-CognitiveLoad\data"
tobii_dir_name = r"tobii"

dfs = get_subject_dfs(directory, tobii_dir_name)
"""

import os

import pandas as pd

print("\n")


def find_line_starting_with(file_path, start_string):
    """
    Finds the first line number where a line in the file `file_path` starts with the specific string `start_string`.
    Used for finding the start of the data and omitting the header in a file.
    """
    line_number = -1

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for number, line in enumerate(file, start=1):
                # Split the line by tab and check the first element
                if line.startswith(start_string):
                    line_number = number
                    break
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    if line_number < 0:
        line_number = 0

    return line_number


def tracker_data_from_tsv_to_df(
    file_path, signal_start=None, sep="\t", search_for_signal_start=True
):
    """
    Reads the eye tracker data from a tsv file, skips the header (skips everything until "TimeStamp" is reached) and returns a pandas dataframe.
    """

    if search_for_signal_start and signal_start is None:
        signal_start = find_line_starting_with(file_path, "TimeStamp")

    # returns a pandas dataframe of eye tracker data from a tsv file
    try:
        if signal_start is not None:
            df = pd.read_csv(file_path, sep=sep, skiprows=signal_start - 1)
        else:
            df = pd.read_csv(file_path, sep=sep)
    except Exception as e:
        print(f"Exception: {e}")
        print("Returning an empty dataframe.")
        df = pd.DataFrame({})
    return df


def get_dfs(source_dir, sep="\t", tobii_dir_name=None, search_for_signal_start=True):
    """
    Expects tobii data in tsv file.

    Parameters:
    - source_dir: string - the directory containing the tobii data
    - tobii_dir_name: string - the name of the directory containing the tobii data (optional - makes sure we don't read other tsv data)

    Returns: dictionary of subject dataframes. Keys are filenames.
    """
    # Returns a dictionary of subject dataframes from the source directory
    # The keys are the subject names and the values are the dataframes
    dfs = {}

    for root, dirs, files in os.walk(source_dir):
        # check files in the root directory
        for file in files:
            if sep == "\t" and file.endswith(".tsv"):
                print(f"Reading file: {file}")
                tobii_file_path = os.path.join(root, file)
                if search_for_signal_start:
                    signal_start = find_line_starting_with(
                        tobii_file_path,
                        "TimeStamp",  # "TimeStamp" for the original data in tobii eye tracker
                    )
                    df = tracker_data_from_tsv_to_df(
                        tobii_file_path,
                        signal_start,
                        sep=sep,
                        search_for_signal_start=search_for_signal_start,
                    )
                else:
                    df = tracker_data_from_tsv_to_df(
                        tobii_file_path,
                        None,
                        sep=sep,
                        search_for_signal_start=search_for_signal_start,
                    )
                dfs[file] = df
            elif sep == "," and file.endswith(".csv"):
                print(f"Reading file: {file}")
                tobii_file_path = os.path.join(root, file)
                df = pd.read_csv(tobii_file_path, sep=sep)
                dfs[file] = df

        # recursively check the subdirectories
        for dir in dirs:
            if tobii_dir_name is None or dir == tobii_dir_name:
                dfs.update(get_dfs(os.path.join(root, dir), tobii_dir_name))

    return dfs
