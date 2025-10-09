import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

######################################
########### MAIN FUNCTIONS ###########
######################################


def calculate_saccades_and_fixations(
    df: pd.DataFrame,
    window_start_time: float,
    window_end_time: float,
    algorithm="I-DT",
    dispersion_threshold=100,
    velocity_threshold=30,
    min_fixation_duration=100,
    max_time_gap=400,  # max time gap between two points in a potential fixation, TODO: should be set to maximum blink time
) -> tuple:
    """
    Warning:
    - Assumes time is in milliseconds!
    - Assumes that invalid data is replaced with np.nan or float("nan") values before calling this function.

    Calculate saccades and fixations from Tobii eye tracker data based on the provided `algorithm`.

    Supported algorithms:
    - Velocity Threshold Identification (I-VT)
    - Dispersion-Threshold Identification (I-DT)

    Parameters:
    Default values of the parameters min_fixation_duration and max_time_gap are in milliseconds and velocity_threshold is in pixels/millisecond!
    - df (pd.DataFrame): DataFrame containing Tobii eye tracker data with columns 'TimeStamp', 'GazePointX', 'GazePointY'.
    - window_start_time (float): Start time of the window of interest based on the `TimeStamp` column.
    - window_end_time (float): End time of the window of interest based on the `TimeStamp` column.
    - algorithm (str): Algorithm to use for detecting saccades and fixations. Supported values are 'I-VT' and 'I-DT'.
    - dispersion_threshold (float): Threshold for detecting saccades. Ignored if algorithm is 'I-VT'.
    - velocity_threshold (float): Threshold for detecting saccades. Ignored if algorithm is 'I-DT'.
    - min_fixation_duration (int): Minimum duration for a fixation in milliseconds.
    - max_time_gap (int): Maximum time gap between two consecutive points in milliseconds. If the time gap is greater than this value, the current fixation/saccade is ended. It should probably be set to maximum blink time (at least for fixations because fixation can continue after a blink).

    TODO: make the return values consistent for both algorithms.
    Returns: Depends on the `algorithm`:
    - if I-VT:
        - saccades (list): List of saccades with start and end timestamps. [start_time, end_time, path=[(time, x, y), ...]]
        - fixations (list): List of fixations with start and end timestamps and centroid positions. [fixation_start, fixation_end, fixation_centroid_x, fixation_x_std, fixation_centroid_y, fixation_y_std]
    - if I-DT:
        - saccades: None
        - fixations (dict): Dictionary with features of fixations

    Saccade rules of this implementation:
    1. A saccade must be preceded by a fixation.
    2. A saccade must be followed by a fixation.
    3. A saccade must wholly lie within the TOI (Time of Interest).

    Fixation rules of this implementation:
    1. Fixation must be at least `min_fixation_duration` long.
    2. Fixation does not necessarily wholly lie within the TOI i.e., we may start the first fixation at the start of the TOI and we may end the last fixation at the end of the TOI, although it may be incomplete. (Note: this is implemented for i_dt(), for i_vt() we drop the last fixation if it is incomplete)
    """

    if algorithm == "I-DT":
        saccades, fixations = i_dt(
            df,
            window_start_time,
            window_end_time,
            max_time_gap,
            dispersion_threshold=dispersion_threshold,
            min_fixation_duration=min_fixation_duration,
        )
    elif algorithm == "I-VT":
        saccades, fixations = i_vt(
            df,
            window_start_time,
            window_end_time,
            max_time_gap,
            velocity_threshold=velocity_threshold,
            min_fixation_duration=min_fixation_duration,
        )
    else:
        raise ValueError(
            "Unsupported algorithm. Supported values are 'I-VT' and 'I-DT'."
        )

    return saccades, fixations


def i_vt(
    df,
    window_start_time,
    window_end_time,
    max_time_gap,
    velocity_threshold,
    min_fixation_duration,
):
    """
    I-VT algorithm for detecting saccades and fixations from Tobii eye tracker data.

    Currently, the velocity means the velocity of gaze on the screen in pixels per millisecond. We do not calculate angular velocity because we do not have the distance from the screen.

    TODO: Implement parameter distance_from_screen: list and if it is not None, calculate angular velocity.
    """

    # 0. Filter the data based on the window start and end times and drop nan rows because we cannot calculate velocity for them
    df = df.loc[df["TimeStamp"].between(window_start_time, window_end_time)].copy()
    df.dropna(subset=["GazePointX", "GazePointY"], inplace=True)

    # 1. Calculate time differences and velocities between consecutive points - both are one datapoint shorter than other data!
    df["time_diff"] = df["TimeStamp"].diff()

    # TODO - check if this func works properly
    df["velocity"] = np.append(
        [0], getVelocities(df["GazePointX"], df["GazePointY"], df["TimeStamp"])
    )

    # 2. Detect saccades and fixations based on the velocity threshold (I-VT algorithm)
    saccades = []
    fixations = []
    fixation_start = None  # None means there is no fixation currently
    cur_saccade = []
    prev_i = None  # index of the previous row because we dropped some rows inbetween due to missing data
    preceded_by_fixation = (
        False  # Flag to check if the current saccade is preceded by a fixation
    )

    for i, row in df.iterrows():
        if prev_i is not None:
            fixation_end = df.loc[prev_i]["TimeStamp"]  # possible fixation-end time

        # Check if the time gap is too big - should we end the current fixation/saccade or drop it?
        # Currently, we drop it. We want every saccade to be preceded and followed by a fixation.
        if row["time_diff"] > max_time_gap:
            # drop the current fixation/saccade
            fixation_start = None
            preceded_by_fixation = False
            cur_saccade = []

            """
            Version where we end the current fixation/saccade instead of dropping it:

            if fixation_start is not None:
                # End of (possible) fixation.
                preceded_by_fixation = endFixation(
                    fixations,
                    fixation_start,
                    fixation_end,
                    df,
                    prev_i,
                    min_fixation_duration,
                )
                fixation_start = None

            else:
                fixation_start = endSaccade(
                    row, cur_saccade, saccades, preceded_by_fixation
                )
                cur_saccade = []  # Reset the saccade list
                preceded_by_fixation = False  # because too big of a gap in timestamps
            """

        # Check if the velocity is greater than the threshold
        elif row["velocity"] > velocity_threshold:
            if fixation_start is not None:
                preceded_by_fixation = endFixation(
                    fixations,
                    fixation_start,
                    fixation_end,
                    df,
                    prev_i,
                    min_fixation_duration,
                )

                fixation_start = None

            # Add the current point to the current saccade if a fixation already happened before else ignore
            if preceded_by_fixation:
                if cur_saccade == []:
                    # Add fixation point to the saccade TODO: should we add the last fixation point to the saccade?
                    first_row = df.loc[prev_i]
                    cur_saccade.append(
                        (
                            first_row["TimeStamp"],
                            first_row["GazePointX"],
                            first_row["GazePointY"],
                            # first_row["velocity"], - TODO - think which rows' velocities to add!
                        )
                    )

                # Building the saccade path
                cur_saccade.append(
                    (
                        row["TimeStamp"],
                        row["GazePointX"],
                        row["GazePointY"],
                        # row["velocity"],
                    )
                )

        # Else, we are in a (middle of or at the start of a) fixation
        else:
            if fixation_start is None:
                fixation_start = endSaccade(
                    row, cur_saccade, saccades, preceded_by_fixation
                )
                cur_saccade = []  # Reset the saccade list
            else:
                preceded_by_fixation = True  # Continue the current fixation

        prev_i = i

    # DO NOT end the last fixation/saccade because the data might be incomplete
    return saccades, fixations


def i_dt(
    df,
    window_start_time,
    window_end_time,
    max_time_gap,
    dispersion_threshold,
    min_fixation_duration,
):
    """
    I-DT algorithm for detecting fixations from Tobii eye tracker data.

    Returns: None, fixations (dict) - dict with features of fixations

    """

    def update_fixations(fixations, still_points):
        features = [
            "Fixation_StartTime",
            "Fixation_EndTime",
            "Fixation_Duration",
            "Fixation_CentroidX",
            "Fixation_CentroidY",
            "Fixation_StdX",
            "Fixation_StdY",
            "Fixation_Dispersion",
            "Fixation_RangeX",
            "Fixation_RangeY",
            "Fixation_MaxX",
            "Fixation_MinX",
            "Fixation_MaxY",
            "Fixation_MinY",
            "Fixation_FirstX",
            "Fixation_FirstY",
            "Fixation_LastX",
            "Fixation_LastY",
        ]

        # add the features to the fixations dictionary
        for feature in features:
            fixations.setdefault(feature, [])

        fix_start = np.min(still_points["TimeStamp"])
        fix_end = np.max(still_points["TimeStamp"])
        fixations["Fixation_StartTime"].append(fix_start)
        fixations["Fixation_EndTime"].append(fix_end)
        fixations["Fixation_Duration"].append(fix_end - fix_start)
        fixations["Fixation_CentroidX"].append(np.mean(still_points["GazePointX"]))
        fixations["Fixation_CentroidY"].append(np.mean(still_points["GazePointY"]))
        fixations["Fixation_StdX"].append(np.std(still_points["GazePointX"]))
        fixations["Fixation_StdY"].append(np.std(still_points["GazePointY"]))

        max_x = max(still_points["GazePointX"])
        min_x = min(still_points["GazePointX"])
        max_y = max(still_points["GazePointY"])
        min_y = min(still_points["GazePointY"])
        fixations["Fixation_Dispersion"].append(dispersion(max_x, min_x, max_y, min_y))
        fixations["Fixation_RangeX"].append(max_x - min_x)
        fixations["Fixation_RangeY"].append(max_y - min_y)
        fixations["Fixation_MaxX"].append(max_x)
        fixations["Fixation_MinX"].append(min_x)
        fixations["Fixation_MaxY"].append(max_y)
        fixations["Fixation_MinY"].append(min_y)

        # save the first and last points of the fixation
        first_x = still_points["GazePointX"][0]
        first_y = still_points["GazePointY"][0]
        last_x = still_points["GazePointX"][-1]
        last_y = still_points["GazePointY"][-1]
        fixations["Fixation_FirstX"].append(first_x)
        fixations["Fixation_FirstY"].append(first_y)
        fixations["Fixation_LastX"].append(last_x)
        fixations["Fixation_LastY"].append(last_y)

        return fixations

    # Get only the data in the window of interest
    df_in_range = (
        df.loc[
            (df["TimeStamp"] >= window_start_time)
            & (df["TimeStamp"] <= window_end_time)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # drop rows with missing data
    df_in_range.dropna(subset=["GazePointX", "GazePointY"], inplace=True)

    # Dict with features of fixations to return
    fixations = {}

    # Points of a potential fixation
    still_points = {"TimeStamp": [], "GazePointX": [], "GazePointY": []}

    # Iterate over the rows of the DataFrame
    row_indx = 0
    while row_indx < df_in_range.shape[0]:
        row = df_in_range.iloc[row_indx]

        if len(still_points["TimeStamp"]) > 0:
            # Happens when the previous still_points were not enough to declare a fixation, so the first point was popped
            max_x = max(still_points["GazePointX"])
            min_x = min(still_points["GazePointX"])
            max_y = max(still_points["GazePointY"])
            min_y = min(still_points["GazePointY"])
            disp = dispersion(max_x, min_x, max_y, min_y)
        else:
            # initialize variables
            disp = 0  # dispersion
            max_x = None
            max_y = None
            min_x = None
            min_y = None

        # Add still/fixation points to still_points until dispersion_threshold is reached
        while row_indx < df_in_range.shape[0] and disp < dispersion_threshold:
            row = df_in_range.iloc[row_indx]  # update row
            t = row["TimeStamp"]
            if (
                still_points["TimeStamp"] != []
                and (t - still_points["TimeStamp"][-1]) > max_time_gap
            ):
                # if the time gap is too big, end the current (potential) fixation
                break

            x = row["GazePointX"]
            y = row["GazePointY"]

            max_x = max(max_x, x) if max_x is not None else x
            min_x = min(min_x, x) if min_x is not None else x
            max_y = max(max_y, y) if max_y is not None else y
            min_y = min(min_y, y) if min_y is not None else y

            # calculate dispersion: (max_x - min_x) + (max_y - min_y)
            disp = dispersion(max_x, min_x, max_y, min_y)
            # print("   Dispersion:", disp)

            # if dispersion thresh is not reached, add the point to still_points (fixation)
            if disp < dispersion_threshold:
                still_points["TimeStamp"].append(t)
                still_points["GazePointX"].append(x)
                still_points["GazePointY"].append(y)
                row_indx += 1

        # Dispersion threshold or max_time_gap was reached.
        if row_indx < df_in_range.shape[0] and still_points["TimeStamp"] != []:
            fix_start = np.min(still_points["TimeStamp"])
            fix_end = np.max(still_points["TimeStamp"])
            dt = fix_end - fix_start  # delta time

            # if the window is long enough, declare a fixation
            if dt >= min_fixation_duration:
                # Declare a fixation
                fixations = update_fixations(fixations, still_points)

                # Reset still_points
                still_points = {"TimeStamp": [], "GazePointX": [], "GazePointY": []}
            else:
                # Window is not long enough to declare a fixation
                # Remove the first point and continue
                # print("   Still points before:", still_points)
                for key in still_points:
                    still_points[key].pop(0)  # pop the first point from each list
                # print("   Still points after:", still_points)

        row_indx += 1

    # END THE LAST FIXATION, ALTHOUGH IT MAY BE INCOMPLETE
    if still_points["TimeStamp"] != []:
        fix_start = np.min(still_points["TimeStamp"])
        fix_end = np.max(still_points["TimeStamp"])
        dt = fix_end - fix_start  # delta time

        # if the window is long enough, declare a fixation
        if dt >= min_fixation_duration:
            # Declare a fixation
            fixations = update_fixations(fixations, still_points)

            # Reset still_points
            still_points = {"TimeStamp": [], "GazePointX": [], "GazePointY": []}
        else:
            # Window is not long enough to declare a fixation
            # Remove the first point and continue
            # print("   Still points before:", still_points)
            for key in still_points:
                still_points[key].pop(0)  # pop the first point from each list
            # print("   Still points after:", still_points)

    saccades = None  # saccades will be calcualted later because we need blink data (accompanied with fixations)

    return saccades, fixations


def get_fixations_from_bools(df) -> dict:
    """
    Returns: fixations (dict) - dict with features of fixations. We simply take out the features starting with "Fixation_" from the df and return them as a dict.
    """

    fixations = {}
    for col in df.columns:
        if "Fixation_" in col:
            fixations[col] = df[col].dropna().to_numpy()

    return fixations


def calculate_saccades_from_idt_fixations(
    df,
    idt_fixations,
    blinks,
    window_start_time,
    window_end_time,
    max_saccade_duration,
):
    """
    Warning: ugly code...

    Parameters:
    - idt_fixations (dict): dictionary with features of fixations (returned from i_dt function)
    - blinks (list): list of blinks as tuples of (start_time, end_time)
    - window_start_time (float): Start time of the window of interest based on the `TimeStamp` column.
    - window_end_time (float): End time of the window of interest based on the `TimeStamp` column.

    Returns: saccades (list): List of saccades with start and end timestamps. [start_time, end_time, path=[(time, x, y), ...]]

    Sacade rules:
    - A saccade can durate at most `max_saccade_duration`.
    - A saccade must be preceded by a fixation.
    - A saccade must be followed by a fixation.
    - A saccade must wholly lie within the TOI (Time of Interest).
    - A saccade cannot coexist with a blink. TODO: think if this is necessary and correct the implementation if this rule can be excluded.
    """

    def checkPotentialSaccadeInRegion(
        start, end, blinks, blinks_indx, undefined_regions
    ) -> tuple[bool, int]:
        """
        Checks the undefined region between two fixations for possible blinks and omits them from undefined_regions. Also checks if the current fixation is in the window of interest and if the duration is < `max_saccade_duration`.

        Returns:
        - brk (bool): True if the current fixation is not in the window of interest [window_start_time, window_end_time], False otherwise.
        - blinks_indx (int): Index of the current blink in the list of blinks.
        """

        if end <= start:
            return False, blinks_indx

        # skip if we are not yet in the window of interest
        if start < window_start_time:
            return False, blinks_indx
        # break if we came to the end of the window of interest
        if end > window_end_time:
            return True, blinks_indx

        # check for blinks in the region
        while blinks_indx < len(blinks) and blinks[blinks_indx][1] <= start:
            # skip possible blinks that are before the region
            blinks_indx += 1

        # Rule: A saccade cannot coexist with a blink
        if blinks_indx < len(blinks) and blinks[blinks_indx][0] < end:
            blink_start = blinks[blinks_indx][0]
            if 0 < blink_start - start < max_saccade_duration:
                undefined_regions.append((start, blink_start))

            blink_end = blinks[blinks_indx][1]
            return checkPotentialSaccadeInRegion(
                blink_end, end, blinks, blinks_indx, undefined_regions
            )

        # Rule: A saccade can durate at most `max_saccade_duration`
        elif 0 < end - start < max_saccade_duration:
            undefined_regions.append((start, end))

        return False, blinks_indx

    if idt_fixations == {}:
        return []

    # Get the time windows `undefined_regions` where neither blink nor fixation is happening
    # We will calculate saccades in these windows
    undefined_regions = []
    blinks_indx = 0
    brk = False
    # get possible saccade times considering a saccade must be between two fixations and in absence of blinks
    for i in range(len(idt_fixations["Fixation_StartTime"]) - 1):
        start = idt_fixations["Fixation_EndTime"][i]
        end = idt_fixations["Fixation_StartTime"][i + 1]

        brk, blinks_indx = checkPotentialSaccadeInRegion(
            start, end, blinks, blinks_indx, undefined_regions
        )

        if brk:
            break

    if undefined_regions == []:
        return []

    # Calculate saccades in the `undefined_regions`
    saccades = []
    saccade_path = []
    ur_indx = 0  # undefined region index
    ur = undefined_regions[ur_indx]  # current undefined region
    for i, row in df.iterrows():
        # we are done with the undefined regions
        if ur_indx >= len(undefined_regions):
            break

        # skip rows that are not yet in the region
        if row["TimeStamp"] < ur[0]:
            continue
        # add rows in the region
        elif row["TimeStamp"] <= ur[1]:
            saccade_path.append(
                (row["TimeStamp"], row["GazePointX"], row["GazePointY"])
            )
            continue
        # end of the region
        else:
            saccades.append((ur[0], ur[1], saccade_path))
            saccade_path = []

            ur_indx += 1
            if ur_indx < len(undefined_regions):
                ur = undefined_regions[ur_indx]
            else:
                break

    return saccades


def calculate_saccades_from_bools(df, saccade_bools) -> list:
    """
    Returns:
    - saccades (list): List of saccades with start and end timestamps. [start_time, end_time, path=[(time, x, y), ...]]
    """
    starts_mask = saccade_bools & ~(
        saccade_bools.shift(1).fillna(False).infer_objects()
    )
    ends_mask = saccade_bools & ~(saccade_bools.shift(-1).fillna(False).infer_objects())
    sac_starts = df.loc[starts_mask, "TimeStamp"]
    sac_ends = df.loc[ends_mask, "TimeStamp"]

    saccades = []
    for start, end in zip(sac_starts, sac_ends):
        saccade_path = df.loc[
            (start <= df["TimeStamp"]) & (df["TimeStamp"] <= end),
            ["TimeStamp", "GazePointX", "GazePointY"],
        ].to_numpy()
        saccades.append((start, end, saccade_path))

    return saccades


######################################
######## Get Fix./Sac. Features ######
######################################


def getSaccadeFeatures(saccades, start_time, end_time):
    """
    Returns: Dict[str, float] of saccade features for classification:
    - mean duration
    - std duration
    - duration range (max - min)
    - total duration
    - duration kurtosis
    - duration skewness
    - mean velocity
    - std velocity
    - velocity range (max - min)
    - velocity kurtosis
    - velocity skewness
    - mean amplitude (distance between start and end point)
    - std amplitude
    - amplitude range (max - min)
    - frequency (number of saccades per second)
    - (count) -> don't use this for now, frequency is better if we have different TOI lengths

    Parameters:
    - saccades (list): [start_time, end_time, path=[(time, x, y), ...]] - this is the output of the calculate_saccades_and_fixations function
    """

    features = {}

    # Get durations, velocities and amplitudes
    # velocity and distance are calculated based on start and end coordinates. They could also be calculated as mean velocity/distance between points on saccade path (which we already calculated in the main function i_vt_saccades_fixations)
    # TODO: check if works
    durations = []
    velocities = []
    amplitudes = []
    for sac_start_time, sac_end_time, path in saccades:
        durations.append(sac_end_time - sac_start_time)

        x = [path[0][1], path[-1][1]]
        y = [path[0][2], path[-1][2]]
        t = [sac_start_time, sac_end_time]
        v = getVelocities(x, y, t)[0]
        velocities.append(v)

        d = getDistances(x, y)[0]
        amplitudes.append(d)

    features["Saccade_mean_duration"] = np.mean(durations)
    features["Saccade_std_duration"] = np.mean(durations)
    features["Saccade_duration_range"] = np.max(durations) - np.min(durations)
    features["Saccade_total_duration"] = np.sum(durations)
    features["Saccade_kurtosis_duration"] = kurtosis(durations)
    features["Saccade_skewness_duration"] = skew(durations)

    features["Saccade_mean_velocity"] = np.mean(velocities)
    features["Saccade_std_velocity"] = np.std(velocities)
    features["Saccade_velocity_range"] = np.max(velocities) - np.min(velocities)
    features["Saccade_kurtosis_velocity"] = kurtosis(velocities)
    features["Saccade_skewness_velocity"] = skew(velocities)

    features["Saccade_mean_amplitude"] = np.mean(amplitudes)
    features["Saccade_std_amplitude"] = np.std(amplitudes)
    features["Saccade_amplitude_range"] = np.max(amplitudes) - np.min(amplitudes)

    features["Saccade_frequency"] = len(saccades) / (end_time - start_time)
    features["Saccade_count"] = len(saccades)

    return features


def getFixationFeatures(fixations, start_time, end_time):
    """
    Returns: Dict[str, float] of fixation features for classification:
    - mean duration
    - std duration
    - duration range (max - min)
    - total duration
    - duration kurtosis
    - duration skewness
    - frequency (number of fixations per second)
    - count

    Parameters:
    - fixations (dict): dictionary with features of fixations (returned from i_dt function)
    """

    if fixations == {}:
        return {}

    features = {}

    for key, value in fixations.items():
        if "Time" in key or "Fixation_bool" == key:
            continue
        if len(value) == 0:
            features[key + "_mean"] = np.nan
            features[key + "_std"] = np.nan
            features[key + "_min"] = np.nan
            features[key + "_max"] = np.nan
            features[key + "_range"] = np.nan
            features[key + "_kurtosis"] = np.nan
            features[key + "_skewness"] = np.nan
            continue

        features[key + "_mean"] = np.nanmean(value)
        features[key + "_std"] = np.nanstd(value)
        features[key + "_min"] = np.nanmin(value)
        features[key + "_max"] = np.nanmax(value)
        features[key + "_range"] = np.nanmax(value) - np.nanmin(value)
        features[key + "_kurtosis"] = kurtosis(value)
        features[key + "_skewness"] = skew(value)

    duration = fixations["Fixation_EndTime"][-1] - fixations["Fixation_StartTime"][0]
    if duration == 0:
        return {}
    features["Fixation_frequency"] = len(fixations["Fixation_StartTime"]) / duration
    features["Fixation_count"] = len(fixations["Fixation_StartTime"])

    return features


######################################
# Helper functions
######################################


def endFixation(
    fixations, fixation_start, fixation_end, df, prev_i, min_fixation_duration
):
    """
    End the current fixation and append it to the list of fixations `fixations` if it meets the minimum duration requirement.

    Parameters:
    fixations (list): List to store the detected fixations.
    fixation_start (float): Timestamp when the current fixation started.
    df (pd.DataFrame): DataFrame containing Tobii eye tracker data with columns 'TimeStamp', 'GazePointX', 'GazePointY'.
    prev_i (int): Index of the previous row in the DataFrame.
    min_fixation_duration (int): Minimum duration for a fixation in milliseconds.

    Assumptions:
    - The DataFrame `df` contains consecutive rows of eye-tracking data. Invalid rows should be removed before calling this function.
    - The 'TimeStamp' column in `df` is in milliseconds.
    - The 'GazePointX' and 'GazePointY' columns in `df` contain the x and y coordinates of the gaze points, respectively.

    Returns: preceded_by_fixation (bool): True if fixation is detected, False otherwise.
    """

    # End of the current fixation

    duration = fixation_end - fixation_start
    # print("Duration (ms):", duration)
    if duration >= min_fixation_duration:
        fixation_x = df.loc[
            (fixation_start <= df["TimeStamp"]) & (df["TimeStamp"] <= fixation_end),
            "GazePointX",
        ]
        fixation_centroid_x = np.nanmean(fixation_x)
        fixation_x_std = np.nanstd(fixation_x)

        fixation_y = df.loc[
            (fixation_start <= df["TimeStamp"]) & (df["TimeStamp"] <= fixation_end),
            "GazePointY",
        ]
        fixation_centroid_y = fixation_y.mean()
        fixation_y_std = fixation_y.std()

        fixation = (
            fixation_start,
            fixation_end,
            fixation_centroid_x,
            fixation_x_std,
            fixation_centroid_y,
            fixation_y_std,
        )

        fixations.append(fixation)

        return True
    return False


def endSaccade(row, cur_saccade, saccades, preceded_by_fixation):
    """
    End the current saccade (if there is one) and append it to the list of saccades `saccades`.

    Returns the start of the new fixation.

    Parameters:
    - row (pd.Series): Current row of the DataFrame containing Tobii eye tracker data.
    - cur_saccade (list): List of points that belong to the current saccade.
    - saccades (list): List to store the detected saccades in format [start_time, end_time, path=[(time, x, y), ...]]
    """
    # print("End of (possible) saccade, start of new fixation.")
    # Start of a new fixation
    fixation_start = row["TimeStamp"]

    # End of the current saccade
    if len(cur_saccade) > 0 and preceded_by_fixation:
        sac_start_time = cur_saccade[0][0]
        sac_end_time = fixation_start
        sac_path = cur_saccade + [(sac_end_time, row["GazePointX"], row["GazePointY"])]
        saccades.append((sac_start_time, sac_end_time, sac_path))

    return fixation_start


def getDistances(x, y):
    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)


def getVelocities(x, y, times):
    """
    Parameters:
    - x, y (pd.Series): coordinates
    - times (pd.Series): timestamps

    Returns numpy array of velocities. Length of the list is one less than input parameters.
    """

    return getDistances(x, y) / np.diff(times)


def dispersion(max_x, min_x, max_y, min_y):
    return (max_x - min_x) + (max_y - min_y)


######################################
# VISUALISATION FUNCTIONS
######################################

FIGSIZE = (19.20 * 0.5, 10.80 * 0.5)


def plotSaccades(plt, saccades, fixations, add_text=False):
    for saccade in saccades:
        saccade_path = saccade[2]
        x = [point[1] for point in saccade_path]
        y = [point[2] for point in saccade_path]
        timestamps = [point[0] for point in saccade_path]

        # Find the fixation that corresponds to the start of the saccade
        start_time = saccade[0]
        fixation_color = None
        for fixation in fixations:
            if fixation[0] <= start_time <= fixation[1]:
                fixation_color = (
                    plt.gca().collections[fixations.index(fixation)].get_facecolor()[0]
                )
                break

        for i in range(len(x) - 1):
            plt.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=fixation_color,
            )
            if add_text:
                plt.text(
                    x[i],
                    y[i],
                    timestamps[i],
                    fontsize=7,
                    ha="center",
                )
        if add_text:
            plt.text(
                x[len(x) - 1],
                y[len(x) - 1],
                timestamps[len(x) - 1],
                fontsize=7,
                ha="center",
            )


def visualiseSaccades(saccades, fixations):
    plt.figure(figsize=FIGSIZE)
    plotSaccades(plt, saccades, fixations)
    plt.xlabel("GazePointX")
    plt.ylabel("GazePointY")
    plt.show()


def plotFixations(plt, fixations, add_text=False):
    for fixation in fixations:
        fixation_start = fixation[0]
        fixation_end = fixation[1]
        fixation_centroid_x = fixation[2]
        fixation_x_std = fixation[3]
        fixation_centroid_y = fixation[4]
        fixation_y_std = fixation[5]
        radius = (
            fixation_x_std + fixation_y_std
        ) / 2  # Average of the standard deviations
        plt.scatter(
            fixation_centroid_x,
            fixation_centroid_y,
            s=(radius**2),
            alpha=0.5,
        )  # s is the area of the marker
        if add_text:
            plt.text(
                fixation_centroid_x,
                fixation_centroid_y,
                f"({fixation_start:.0f}, {fixation_end:.0f})",
                fontsize=7,
                ha="center",
            )


def visualiseFixations(fixations):
    plt.figure(figsize=FIGSIZE)
    plotFixations(plt, fixations)
    plt.xlabel("GazePointX")
    plt.ylabel("GazePointY")
    plt.show()


def visualiseSaccadesAndFixations(saccades, fixations, df):
    plt.figure(figsize=FIGSIZE)
    plotFixations(plt, fixations)
    plotSaccades(plt, saccades, fixations)
    plt.xlabel("GazePointX")
    plt.ylabel("GazePointY")
    visualiseRawGazeData(df, its_own_plot=False)
    plt.show()


def visualiseRawGazeData(df, its_own_plot=True):
    if its_own_plot:
        plt.figure(figsize=FIGSIZE)
        plt.xlabel("GazePointX")
        plt.ylabel("GazePointY")

    plt.plot(df["GazePointX"], df["GazePointY"])

    if its_own_plot:
        plt.show()
