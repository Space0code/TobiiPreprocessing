# TobiiPreprocessing

## Eye-Tracking Data Preprocessing Pipeline
This preprocessing was designed for raw CSV-format data obtained with Tobii Pro Spark. It is used in pipelines developed under the [Trust-ME](https://dis.ijs.si/trust-me/) project.

### 1. **Raw Data Structure**
The raw data consists of (55 x ~12MB) TSV files from Tobii eye trackers with the following key columns:
- `TimeStamp`: Time in milliseconds 
- `GazePointX/Y`: Screen coordinates of gaze position
- `ValidityLeft/Right`: Data validity flags (0=invalid, 1=valid)
- `PupilSizeLeft/Right`: Pupil diameter measurements
- `PupilValidityLeft/Right`: Pupil measurement validity flags
- `Event`: Event markers (empty, later dropped)

Invalid data points are marked with -1.0 values and validity flags of 0.

### 2. **Data Loading and Initial Cleaning**
Using tobii_tsv_to_pandasdf.py:
- Automatically detects and skips header information until "TimeStamp" column (first 7 rows)
- Loads raw TSV files into pandas DataFrames
- Maps filenames to subject IDs using JSON lookup files

**Initial cleaning steps:**
- Sets invalid rows (where both left and right validity = 0) to NaN for all columns except TimeStamp
- Sets eye-specific columns to NaN where corresponding validity flag = 0
- Converts TimeStamp to float type
- Drops meaningless Event column

### 3. **Outlier Detection and Correction**
Using filtering.py with `deal_with_gaze_outliers()`:

**Outlier Identification:**
- Calculates gaze movement amplitude (Euclidean distance between consecutive valid points)
- Uses Z-score threshold (default: 2.0) to identify potential outliers
- Creates 3-point neighborhoods (previous and next 3 points) for context analysis

**Outlier Classification:**
- **Lonely points**: Isolated valid points surrounded by invalid data → Set to NaN
- **Large saccades**: Points that fall between previous and next neighborhood means → Preserved
- **Dynamic overshoots**: Points where amplitude is ≥15x the next amplitude and neighbors are on same side → Preserved  
- **True outliers**: Remaining high-amplitude points → Corrected by averaging valid neighbors

### 4. **Signal Smoothing**

**Gaze Data Smoothing:**
- Applies Savitzky-Golay filter (window=3, polynomial order=1) to GazePointX/Y
- Only smooths valid data points, preserves NaN structure

**Pupil Data Smoothing:**
- Applies stronger Savitzky-Golay filter (window=25, polynomial order=3) to pupil size data
- Note: There was a bug in the code where smoothed right pupil data overwrites left pupil data

### 5. **Eye Movement Event Detection**

**Blinks Detection** (using blinks.py):
- Identifies gaps in valid gaze data between 90-300ms duration
- Creates binary blink markers in DataFrame

**Fixations Detection** (using I-DT algorithm from saccades_and_fixations.py):
- Uses dispersion threshold (100 pixels) and minimum duration (100ms)
- Groups consecutive points within dispersion threshold
- Calculates fixation centroids, standard deviations, and other spatial features
- Handles maximum time gaps of 300ms or 400ms (allows for brief interruptions)

**Saccades Detection:**
- Calculated as intervals between fixations
- Filtered to remove periods during blinks
- Maximum saccade duration: 400ms
- Computes velocity and amplitude features

### 6. **Feature Extraction**

The pipeline extracts three levels of features:

**A. Subject-Level Features** (`tobii_subject_features.tsv`):
- Global statistics across entire recording session
- Blink metrics: count, frequency, duration statistics
- Fixation metrics: duration, spatial distribution, dispersion measures  
- Saccade metrics: duration, velocity, amplitude statistics
- Pupil metrics: size statistics, velocity measures for both eyes

**B. Task-Level Features** (`tobii_task_features.tsv`):
- Same feature types as subject-level but calculated per task/label
- Uses segmentation file to identify task boundaries
- Filters out "EOF" and NaN labels

**C. Window-Level Features** (`tobii_features_2.tsv`):
- Features calculated for sliding windows (3-second windows, 25% overlap, 60Hz sampling)
- Uses segmentation metadata from windows_ijs_usi.csv
- Includes three feature types per window:
  - **Raw features**: Direct calculations from window data
  - **Subject-global features**: Constant subject-level statistics  
  - **Subject-relative features**: Window features minus subject-global features

### 7. **Data Segmentation and Windowing**
- Uses predefined segmentation file (windows_ijs_usi.csv) with task labels
- Creates overlapping 3-second windows with 25% overlap
- Maps DataFrame indices to time windows 
- Handles different cognitive load tasks (central_position, questionnaire, etc.)

### 8. **Final Output**
The final feature dataset contains:
- **~430MB TSV file** with comprehensive feature set
- **~470 features per window** including statistical measures, subject-relative comparisons

This preprocessing pipeline transforms raw, noisy eye-tracking data into a clean, feature-rich dataset suitable for cognitive load analysis while preserving temporal structure and handling missing data appropriately.
