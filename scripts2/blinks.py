import numpy as np
import pandas as pd


def getBlinks(df, min_duration=90, max_duration=300, frequency=60, unit="milliseconds"):
    """
    Get blinks from a dataframe of tobii data.
    Assumes data is recorded at 60Hz, but can be changed with frequency parameter.
    Assumes that the dataframe has a "TimeStamp" column which holds timestamps in unit `unit`.

    Parameters:
    - df (pd.DataFrame): the dataframe of tobii data
    - min_duration (int): the minimum duration of a blink in `unit`
    - max_duration (int): the maximum duration of a blink in `unit`
    - unit (str): the unit of the duration of the blink. Can be "milliseconds", "microseconds", or "seconds"

    Returns: list of blinks as tuples of (start_time, end_time)
    """
    frame_interval = 1000 / frequency # average interval between two frames (default in ms)

    # default frame_interval is in milliseconds
    if unit == "microseconds":
        frame_interval *= 1000
    elif unit == "seconds":
        frame_interval /= 1000
    elif unit != "milliseconds":
        print(f"Warning: unit {unit} is not recognized. Defaulting to milliseconds.")

    blinks = []
    i_prev = None
    for i, row in df.iterrows():  # TODO: vectorize this loop - it is slow
        # skip invalid rows
        if pd.isna(row["GazePointX"]) and pd.isna(row["GazePointY"]):
            continue
        # skip the first row
        if i_prev is None:
            i_prev = i
            continue

        start = df.loc[i_prev]["TimeStamp"] + frame_interval
        end = row["TimeStamp"]
        if min_duration <= (end - start) <= max_duration:
            blinks.append((start, end))
        i_prev = i

    return blinks


def getBlinksVectorized(df, min_duration=90, max_duration=300):
    """
    Get blinks from a dataframe of Tobii data.
    Assumes that the dataframe has a "TimeStamp" column holding timestamps,
    and columns "GazePointX" and "GazePointY" that indicate gaze position.
    
    Parameters:
    - df (pd.DataFrame): the dataframe of Tobii data.
    - min_duration (int): the minimum duration of a blink.
    - max_duration (int): the maximum duration of a blink.
    
    Returns:
    - list of tuples: each tuple is (start_time, end_time) for a valid blink.
    """
    # Check required columns
    required_columns = ["GazePointX", "GazePointY", "TimeStamp"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")
    
    if df.empty:
        return []
    
    # Identify rows where both GazePointX and GazePointY are NaN.
    invalid_mask = df["GazePointX"].isna() & df["GazePointY"].isna()
    
    # Get positions of invalid and valid rows (these are positional indices)
    invalid_positions = np.where(invalid_mask)[0]
    valid_positions = np.where(~invalid_mask)[0]
    
    if valid_positions.size == 0 or invalid_positions.size == 0:
        return []
    
    # Remove invalid positions that occur before the first valid row.
    first_valid_position = valid_positions[0]
    last_invalid_position = invalid_positions[-1]
    invalid_positions = invalid_positions[invalid_positions > first_valid_position]
    
    # Group consecutive invalid positions to determine blink segments.
    starts, ends = [], []
    group_start = None
    prev_pos = None
    for pos in invalid_positions:
        if group_start is None:
            # Set group start to the frame before the first invalid frame, if possible.
            if pos > 0:
                group_start = pos - 1
        else:
            # If the current invalid row is not consecutive, finish the current group.
            if prev_pos is not None and pos != prev_pos + 1:
                if prev_pos <= last_invalid_position:
                    starts.append(group_start)
                    ends.append(prev_pos)
                group_start = pos - 1 if pos > 0 else None
        prev_pos = pos

    # Append the final group if it exists.
    if group_start is not None and prev_pos is not None:
        if prev_pos <= last_invalid_position:
            starts.append(group_start)
            ends.append(prev_pos)
    
    # Map positional indices to original index labels.
    original_labels = df.index.to_numpy()
    
    valid_blinks = []
    for start_pos, end_pos in zip(starts, ends):
        if start_pos >= 0 and end_pos < len(df):
            start_label = original_labels[start_pos]
            end_label = original_labels[end_pos]
            start_time = df.loc[start_label, "TimeStamp"]
            end_time = df.loc[end_label, "TimeStamp"]
            duration = end_time - start_time
            if min_duration <= duration <= max_duration:
                valid_blinks.append((start_time, end_time))
    
    return valid_blinks


def calculate_blinks_from_bools(df, bools) -> list[tuple[float, float]]:
    """
    Returns: list of blinks as tuples of (start_time, end_time)
    """
    starts_mask = bools & ~(bools.shift(1).fillna(False).infer_objects())
    ends_mask = bools & ~(bools.shift(-1).fillna(False).infer_objects())
    starts = df.loc[starts_mask, "TimeStamp"].values
    ends = df.loc[ends_mask, "TimeStamp"].values
    return list(zip(starts, ends))


def getBlinkFeatures(blinks, start_time, end_time):
    """
    Calculate blink features for a window of time.

    Parameters:
    - blinks (list): list of blinks as tuples of (start_time, end_time)
    - start_time (float): the start time of the window
    - end_time (float): the end time of the window

    Returns: Dict[str, float] of features:
    - Blink_count: the number of blinks in the window
    - Blink_frequency: the number of blinks per second in the window
    - Blink_duration_sum: the total duration of all blinks in the window
    - Blink_duration_mean: the average duration of a blink in the window
    - Blink_duration_std: the standard deviation of the duration of a blink in the window
    """

    # durations of blinks in the (start_time, end_time) interval (in seconds)
    blink_durations = [
        (blink[1] - blink[0])
        for blink in blinks
        if blink[0] >= start_time and blink[1] <= end_time
    ]

    # window duration in seconds
    window_duration = end_time - start_time

    return {
        "Blink_count": len(blinks),
        "Blink_frequency": len(blinks) / window_duration,
        "Blink_duration_sum": np.sum(blink_durations),
        "Blink_duration_mean": np.mean(blink_durations),
        "Blink_duration_std": np.std(blink_durations),
    }
