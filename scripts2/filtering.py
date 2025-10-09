import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
HELPER FUNCTIONS 
"""


def distanceSquared(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def potential_outliers(df: pd.DataFrame, z_scores_thresh: float) -> np.array:
    # df = df.copy() --> unnecessary and time consuming

    # calculate amplitude if the df does not have it
    if "Amplitude" not in df.columns:
        df_amp = gaze_movement_amplitude(df)
        df = pd.merge(df, df_amp, on="TimeStamp", how="left")

    z_scores = np.abs(df["Amplitude"] - df["Amplitude"].mean()) / df["Amplitude"].std()

    potentials = np.where(np.isnan(z_scores), np.nan, z_scores > z_scores_thresh)
    return potentials


def gaze_movement_amplitude(
    df: pd.DataFrame, amplitude_time_thresh=100
) -> pd.DataFrame:
    """
    Calculates amplitude of gaze movement in pixels in a slightly specific way. It ignores rows where gaze data is missing and calculates distances between rows before and after the nan values (if there are any nan values). It also ignores rows where the time difference between the current and previous non-man row is greater than amplitude_time_thresh.
    """

    # Calculate distance using Euclidean formula
    # Only calculate for rows where both the current and previous points are not NaN
    valid_mask = df[["GazePointX", "GazePointY"]].notna().all(axis=1)

    # Filter out rows where the time difference is greater than the threshold
    mask = (
        df.loc[valid_mask, "TimeStamp"] - df.loc[valid_mask, "TimeStamp"].shift(1)
    ) < amplitude_time_thresh
    mask = valid_mask & mask
    mask.iloc[0] = valid_mask.iloc[
        0
    ]  # Set the first row to True if it was True in valid_mask

    df["Amplitude"] = np.nan
    validX = df.loc[mask, "GazePointX"]
    validY = df.loc[mask, "GazePointY"]
    valid_prevX = df.loc[mask, "GazePointX"].shift(1)
    valid_prevY = df.loc[mask, "GazePointY"].shift(1)
    df.loc[mask, "Amplitude"] = distance(validX, validY, valid_prevX, valid_prevY)

    return df


############################################
##### MAIN OUTLIER DETECTION FUNCTIONS #####


def deal_with_gaze_outliers(
    df: pd.DataFrame, z_scores_thresh: float, visualisation=False
):
    """
    Custom algorithm for detecting outliers in the gaze data. We calculate potential outliers based on the z scores of the amplitude of gaze movement. Among potential outliers we handle:
    - large saccades that could be mistaken for outliers (preserving them)
    - dynamic overshoots (preserving them)
    - outliers (replacing them with the average of the immediate neighbors)

    Returns a DataFrame with the same structure as the input `df` but with potential outliers replaced with the average of their immediate neighbors or replaced with NaN. If `visualisation` is set to True, the function returns a tuple (DataFrame, dict) where the dict contains lists of TimeStamps of large saccades, dynamic overshoots, outliers and noise outliers.
    """

    def get_large_saccades_mask(df):
        return (
            (
                (df["GazePointX_prev_neigh_mean"] < df["GazePointX"])
                & (df["GazePointX"] < df["GazePointX_next_neigh_mean"])
            )
            | (
                (df["GazePointX_prev_neigh_mean"] > df["GazePointX"])
                & (df["GazePointX"] > df["GazePointX_next_neigh_mean"])
            )
            | (
                (df["GazePointY_prev_neigh_mean"] < df["GazePointY"])
                & (df["GazePointY"] < df["GazePointY_next_neigh_mean"])
            )
            | (
                (df["GazePointY_prev_neigh_mean"] > df["GazePointY"])
                & (df["GazePointY"] > df["GazePointY_next_neigh_mean"])
            )
        )

    def get_dynamic_overshoots_mask(df, overshoot_thresh=15):
        """If the current row's amplitude is at least `overshoot_thersh` times larger than the next amplitude and the previous and next points are on the same X and/or(?) Y side of the current point, then we have a dynamic overshoot.
        The literature says the saccade is usually ~20 times larger than the corresponding overshoot, therefore we set the threshold a bit lower, to 15 by default."""

        amplitude_cond = df["Amplitude"] >= overshoot_thresh * df["Amplitude_next_1"]
        side_cond = (
            (df["GazePointX_prev_1"] < df["GazePointX"])
            & (df["GazePointX_next_1"] < df["GazePointX"])
            | (df["GazePointX_prev_1"] > df["GazePointX"])
            & (df["GazePointX_next_1"] > df["GazePointX"])
            | (df["GazePointY_prev_1"] < df["GazePointY"])
            & (df["GazePointY_next_1"] < df["GazePointY"])
            | (df["GazePointY_prev_1"] > df["GazePointY"])
            & (df["GazePointY_next_1"] > df["GazePointY"])
        )

        return amplitude_cond & side_cond

    if df.empty:
        return df

    df = df.copy()

    df = gaze_movement_amplitude(df)

    # calculate potential outliers using z_score_thresh
    potentials_mask = potential_outliers(df, z_scores_thresh)

    print(
        "percentage of potential outliers is",
        np.nansum(potentials_mask) / len(potentials_mask) * 100,
        "%",
    )

    # Calculate amplitude if not present
    if "Amplitude" not in df.columns:
        df = gaze_movement_amplitude(df)

    # Initialize lists for visualization purposes
    times_large_saccade = []
    times_dynamic_overshoot = []
    times_outlier = []
    times_lonely_point = []

    # Step 1: Create shifted columns for the neighborhoods
    df["GazePointX_prev_1"] = df["GazePointX"].shift(1)
    df["GazePointY_prev_1"] = df["GazePointY"].shift(1)
    df["Amplitude_prev_1"] = df["Amplitude"].shift(1)

    df["GazePointX_prev_2"] = df["GazePointX"].shift(2)
    df["GazePointY_prev_2"] = df["GazePointY"].shift(2)
    df["Amplitude_prev_2"] = df["Amplitude"].shift(2)

    df["GazePointX_prev_3"] = df["GazePointX"].shift(3)
    df["GazePointY_prev_3"] = df["GazePointY"].shift(3)
    df["Amplitude_prev_3"] = df["Amplitude"].shift(3)

    df["GazePointX_next_1"] = df["GazePointX"].shift(-1)
    df["GazePointY_next_1"] = df["GazePointY"].shift(-1)
    df["Amplitude_next_1"] = df["Amplitude"].shift(-1)

    df["GazePointX_next_2"] = df["GazePointX"].shift(-2)
    df["GazePointY_next_2"] = df["GazePointY"].shift(-2)
    df["Amplitude_next_2"] = df["Amplitude"].shift(-2)

    df["GazePointX_next_3"] = df["GazePointX"].shift(-3)
    df["GazePointY_next_3"] = df["GazePointY"].shift(-3)
    df["Amplitude_next_3"] = df["Amplitude"].shift(-3)

    df["GazePointX_prev_neigh_mean"] = df[
        ["GazePointX_prev_1", "GazePointX_prev_2", "GazePointX_prev_3"]
    ].mean(axis=1)
    df["GazePointY_prev_neigh_mean"] = df[
        ["GazePointY_prev_1", "GazePointY_prev_2", "GazePointY_prev_3"]
    ].mean(axis=1)
    df["GazePointX_next_neigh_mean"] = df[
        ["GazePointX_next_1", "GazePointX_next_2", "GazePointX_next_3"]
    ].mean(axis=1)
    df["GazePointY_next_neigh_mean"] = df[
        ["GazePointY_next_1", "GazePointY_next_2", "GazePointY_next_3"]
    ].mean(axis=1)

    # Step 2: Check for lonely points
    # A point is "lonely" if the 3-step neighborhood is NaN
    left_na_mask = (
        df[["GazePointX_prev_1", "GazePointX_prev_2", "GazePointX_prev_3"]]
        .isna()
        .all(axis=1)
    )
    right_na_mask = (
        df[["GazePointX_next_1", "GazePointX_next_2", "GazePointX_next_3"]]
        .isna()
        .all(axis=1)
    )
    lonely_mask = left_na_mask & right_na_mask

    # Handle lonely points
    df.loc[lonely_mask, df.columns != "TimeStamp"] = np.nan
    times_lonely_point.extend(df.loc[lonely_mask, "TimeStamp"].tolist())

    # Step 3: Check for potential outliers that aren't lonely
    potential_outlier_mask = ~lonely_mask & potentials_mask

    # Step 4: Check for large saccades and dynamic overshoots
    large_saccades_mask = get_large_saccades_mask(df)
    dynamic_overshoots_mask = get_dynamic_overshoots_mask(df)

    # Step 5: Handle outliers based on the masks
    outlier_mask = potential_outlier_mask & ~(
        large_saccades_mask | dynamic_overshoots_mask
    )

    # Collect timestamps for visualization purposes
    times_large_saccade.extend(df.loc[large_saccades_mask, "TimeStamp"].tolist())
    times_dynamic_overshoot.extend(
        df.loc[dynamic_overshoots_mask, "TimeStamp"].tolist()
    )
    times_outlier.extend(df.loc[outlier_mask, "TimeStamp"].tolist())

    # Step 6: Update outliers by averaging the neighbors
    left_x = (
        df["GazePointX_prev_1"]
        .combine_first(df["GazePointX_prev_2"])
        .combine_first(df["GazePointX_prev_3"])
    )
    right_x = (
        df["GazePointX_next_1"]
        .combine_first(df["GazePointX_next_2"])
        .combine_first(df["GazePointX_next_3"])
    )

    left_y = (
        df["GazePointY_prev_1"]
        .combine_first(df["GazePointY_prev_2"])
        .combine_first(df["GazePointY_prev_3"])
    )
    right_y = (
        df["GazePointY_next_1"]
        .combine_first(df["GazePointY_next_2"])
        .combine_first(df["GazePointY_next_3"])
    )

    df["GazePointX_mean_neighbors"] = (left_x + right_x) / 2
    df["GazePointY_mean_neighbors"] = (left_y + right_y) / 2

    # we set to average but for rows where left_x is nan, we use right_x and vice versa; if both are nan, we set the value to nan
    df.loc[outlier_mask, "GazePointX"] = (
        df["GazePointX_mean_neighbors"].combine_first(left_x).combine_first(right_x)
    )
    df.loc[outlier_mask, "GazePointY"] = (
        df["GazePointY_mean_neighbors"].combine_first(left_y).combine_first(right_y)
    )

    # Step 7: Cleanup temporary columns
    df.drop(
        columns=[
            "GazePointX_prev_1",
            "GazePointY_prev_1",
            "Amplitude_prev_1",
            "GazePointX_prev_2",
            "GazePointY_prev_2",
            "Amplitude_prev_2",
            "GazePointX_prev_3",
            "GazePointY_prev_3",
            "Amplitude_prev_3",
            "GazePointX_next_1",
            "GazePointY_next_1",
            "Amplitude_next_1",
            "GazePointX_next_2",
            "GazePointY_next_2",
            "Amplitude_next_2",
            "GazePointX_next_3",
            "GazePointY_next_3",
            "Amplitude_next_3",
            "GazePointX_prev_neigh_mean",
            "GazePointY_prev_neigh_mean",
            "GazePointX_next_neigh_mean",
            "GazePointY_next_neigh_mean",
            "GazePointX_mean_neighbors",
            "GazePointY_mean_neighbors",
        ],
        inplace=True,
    )

    if visualisation:
        return df, {
            "large saccade": times_large_saccade,
            "dynamic overshoot": times_dynamic_overshoot,
            "outlier": times_outlier,
            "lonely outlier": times_lonely_point,
        }

    return df


def deal_with_pupil_outliers(
    df: pd.DataFrame, z_scores_thresh: float, visualisation=False
):
    """
    Basic z-scores thresholding for pupil size. We replace the outliers with NaN.
    """

    if df.empty:
        return df

    df = df.copy()

    # Set invalid pupil sizes to NaN; otherwise they ruin the z-scores calculations
    df.loc[df["PupilValidityLeft"] == 0, "PupilSizeLeft"] = np.nan
    df.loc[df["PupilValidityRight"] == 0, "PupilSizeRight"] = np.nan

    # Calculate z-scores
    z_scores_left = (
        np.abs(df["PupilSizeLeft"] - df["PupilSizeLeft"].mean())
        / df["PupilSizeLeft"].std()
    )
    z_scores_right = (
        np.abs(df["PupilSizeRight"] - df["PupilSizeRight"].mean())
        / df["PupilSizeRight"].std()
    )

    # Replace outliers with NaN
    df.loc[z_scores_left > z_scores_thresh, "PupilSizeLeft"] = np.nan
    df.loc[z_scores_right > z_scores_thresh, "PupilSizeRight"] = np.nan

    if visualisation:
        return df, {
            "outliers_left": df.loc[
                z_scores_left > z_scores_thresh, "TimeStamp"
            ].tolist(),
            "outliers_right": df.loc[
                z_scores_right > z_scores_thresh, "TimeStamp"
            ].tolist(),
        }

    return df


############################################
########### VISUALISATIONS #################


def visualise_outlier_detection(
    df_og,
    df1_without_outliers,
    times_to_plot,
    plot_label="outlier detection (plot in filtering.py)",
    range_ms=300,
    num_plots=3,
):
    """
    Plots the amplitude of gaze movement and gaze points before and after outlier detection. The plots are centered around the timestamps from the `times_to_plot` list and deviate from the center for `range_ms` milliseconds.

    Parameters:
    - df_og: original dataframe
    - df1_without_outliers: dataframe with outliers removed (with the function deal_with_outliers() and visualisation=True)
    - times_to_plot: list of timestamps to plot -> you get them from the dict which is output from deal_with_outliers() with visualisation=True
    """

    FIGSIZE = (16, 10)

    # randomly pick at most `num_plots` times to plot from `times_to_plot`
    times_ = random.sample(times_to_plot, min(num_plots, len(times_to_plot)))

    for time in times_:
        df1 = df_og.copy()
        if df1.empty:
            print("Empty dataframe 0")
            continue
        range = [time - range_ms, time + range_ms]
        df1 = df1[df1["TimeStamp"].between(range[0], range[1])]
        df1_without_outliers_ = df1_without_outliers[
            df1_without_outliers["TimeStamp"].between(range[0], range[1])
        ]
        if df1.empty:
            print("Empty dataframe 1")
            continue
        df_amp = gaze_movement_amplitude(df1)

        plt.figure(figsize=FIGSIZE, label=plot_label)
        plt.subplot(2, 1, 1, title="Amplitude of change")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.plot(
            df_amp["TimeStamp"],
            df_amp["Amplitude"],
            label="Amplitude before",
            color="red",
            marker="o",
            linestyle="--",
            alpha=0.5,
        )
        # for i, row in df_amp.iterrows():
        # plt.text(row["TimeStamp"], row["Amplitude"], f'{row["Amplitude"]:.2f}', fontsize=8, ha='right')
        plt.plot(
            df1_without_outliers_["TimeStamp"],
            df1_without_outliers_["Amplitude"],
            label="Amplitude after",
            color="blue",
            marker="o",
            alpha=0.5,
        )
        # for i, row in df1_without_outliers_.iterrows():
        # plt.text(row["TimeStamp"], row["Amplitude"], f'{row["Amplitude"]:.2f}', fontsize=8, ha='right')
        plt.xlabel("Time [ms]")
        plt.xlim(range)
        plt.ylabel("Amplitude of change")
        plt.legend()

        plt.subplot(2, 1, 2, title="Gaze points")
        plt.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.plot(
            df1["TimeStamp"],
            df1["GazePointX"],
            label="X before",
            marker="o",
            alpha=0.5,
            color="red",
            linestyle="--",
        )
        plt.plot(
            df1["TimeStamp"],
            df1["GazePointY"],
            label="Y before",
            marker="o",
            alpha=0.5,
            color="red",
            linestyle="--",
        )
        # for i, row in df1.iterrows():
        # plt.text(row["TimeStamp"], row["GazePointX"], f'{row["GazePointX"]:.2f}', fontsize=8, ha='right')
        # plt.text(row["TimeStamp"], row["GazePointY"], f'{row["GazePointY"]:.2f}', fontsize=8, ha='right')
        plt.plot(
            df1_without_outliers_["TimeStamp"],
            df1_without_outliers_["GazePointX"],
            label="X after",
            marker="o",
            alpha=0.5,
            color="blue",
        )
        # for i, row in df1_without_outliers_.iterrows():
        # plt.text(row["TimeStamp"], row["GazePointX"], f'{row["GazePointX"]:.2f}', fontsize=8, ha='right')
        # plt.text(row["TimeStamp"], row["GazePointY"], f'{row["GazePointY"]:.2f}', fontsize=8, ha='right')

        plt.plot(
            df1_without_outliers_["TimeStamp"],
            df1_without_outliers_["GazePointY"],
            label="Y after",
            marker="o",
            alpha=0.5,
            color="green",
        )
        plt.xlabel("Time [ms]")
        plt.xlim(range)
        plt.ylabel("Pixel X and Y coordinates")
        plt.legend()
        plt.show()
