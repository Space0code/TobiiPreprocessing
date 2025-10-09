def getPupilFeatures(timestamps, sizeLeft, sizeRight):
    """
    Returns the pupillary metrics that can be used as features for classification.

    Calculated features:
    - mean pupil size
    - std of pupil size
    - pupil size variance
    - pupil size skewness
    - pupil size kurtosis
    - pupil size range (max - min)
    - mean velocity of pupil size changes
    - std velocity of pupil size changes
    - range velocity of pupil size changes


    Parameters:
    - timestamps (pd.Series): the timestamps of the tracker data
    - sizeLeft (pd.Series): the pupil size data of the left eye
    - sizeRight (pd.Series): the pupil size data of the right eye

    Returns: Dict[str, float] of features
    """

    features = {}

    # mean pupil size
    features["Pupil_mean_size_left"] = sizeLeft.mean()
    features["Pupil_mean_size_right"] = sizeRight.mean()

    # std of pupil size
    features["Pupil_std_size_left"] = sizeLeft.std()
    features["Pupil_std_size_right"] = sizeRight.std()

    # pupil size variance
    features["Pupil_var_size_left"] = sizeLeft.var()
    features["Pupil_var_size_right"] = sizeRight.var()

    # pupil size skewness
    features["Pupil_skew_size_left"] = sizeLeft.skew()
    features["Pupil_skew_size_right"] = sizeRight.skew()

    # pupil size kurtosis
    features["Pupil_kurt_size_left"] = sizeLeft.kurt()
    features["Pupil_kurt_size_right"] = sizeRight.kurt()

    # pupil size range (max - min)
    features["Pupil_range_size_left"] = sizeLeft.max() - sizeLeft.min()
    features["Pupil_range_size_right"] = sizeRight.max() - sizeRight.min()

    # velocity of pupil size changes
    sizeLeft_diff = sizeLeft.diff()
    sizeRight_diff = sizeRight.diff()
    time_diff = timestamps.diff()
    velocity_left = sizeLeft_diff / time_diff
    velocity_right = sizeRight_diff / time_diff

    # mean velocity of pupil size changes
    features["Pupil_mean_velocity_left"] = (velocity_left).mean()
    features["Pupil_mean_velocity_right"] = (velocity_right).mean()

    # std velocity of pupil size changes
    features["Pupil_std_velocity_left"] = (velocity_left).std()
    features["Pupil_std_velocity_right"] = (velocity_right).std()

    # range velocity of pupil size changes
    features["Pupil_range_velocity_left"] = (velocity_left).max() - (
        velocity_left
    ).min()
    features["Pupil_range_velocity_right"] = (velocity_right).max() - (
        velocity_right
    ).min()

    return features
