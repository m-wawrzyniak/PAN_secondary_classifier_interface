import pandas as pd
from pathlib import Path
from collections import Counter

def smooth_classification(csv_path: Path, window: int, save_path: Path | None = None):
    """
    Smooth is_face classification along time using a majority vote window.

    Parameters
    ----------
    csv_path : Path
        Path to CSV with columns including ['frame', 'is_face']
    window : int
        Odd window size (number of frames) for smoothing. Must be >=1.
        If 0, no smoothing is applied.
    save_path : Path | None
        Where to save the smoothed CSV. If None, overwrites original CSV.
    """
    df = pd.read_csv(csv_path)
    if 'is_face' not in df.columns:
        raise ValueError(f"\t\t'is_face' column not found in {csv_path}")

    if window <= 0:
        print(f"\t\tWindow=0, no smoothing applied for {csv_path.name}")
        return df

    if window % 2 == 0:
        raise ValueError("\t\tWindow size must be odd for symmetric smoothing")

    half_w = window // 2
    smoothed = df['is_face'].tolist()
    n = len(smoothed)

    for i in range(n):
        # define the window around the current frame
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window_values = smoothed[start:end]
        # majority vote
        majority_label = Counter(window_values).most_common(1)[0][0]
        smoothed[i] = majority_label

    df['is_face'] = smoothed

    if save_path is None:
        save_path = csv_path

    df.to_csv(save_path, index=False)
    print(f"\t\t Smoothed classification saved. ({window}-frame window)")

    return df

def prune_face_frames_local(rec_dict, aggregate_csv_path=None):
    """
    Creates a fresh augmented_face_frames.csv for a single recording.
    Optionally merges metadata from a global aggregate CSV (only for this recording),
    then keeps only frames classified as is_face=1.
    """

    face_csv_path = Path(rec_dict["output_dir"]) / "face_frames.csv"
    model_csv_path = Path(rec_dict["output_dir"]) / "model_class.csv"
    augmented_csv_path = Path(rec_dict["output_dir"]) / "augmented_face_frames.csv"

    if not face_csv_path.exists():
        print(f"\t\t! Missing face_frames.csv → {face_csv_path}")
        return
    if not model_csv_path.exists():
        print(f"\t\t! Missing model_class.csv → {model_csv_path}")
        return

    face_df = pd.read_csv(face_csv_path)
    model_df = pd.read_csv(model_csv_path)

    # Construct actual frame names to match CNN predictions
    face_df["frame_name"] = face_df["timestamp [ns]"].astype(str) + "_" + face_df["suffix"].astype(str) + ".jpg"

    # Keep only frames classified as face
    face_frames_set = set(model_df[model_df["is_face"] == 1]["frame"])
    filtered_df = face_df[face_df["frame_name"].isin(face_frames_set)].copy()

    # Merge extra metadata from aggregate CSV if provided
    if aggregate_csv_path:
        aggregate_df = pd.read_csv(aggregate_csv_path)
        # Keep only rows for this recording
        rec_id = rec_dict.get("recording_id")
        if "recording_id" in aggregate_df.columns:
            agg_rec_df = aggregate_df[aggregate_df["recording_id"] == rec_id]
            # Merge on timestamp + suffix
            filtered_df = pd.merge(
                filtered_df,
                agg_rec_df,
                how="left",
                on=["timestamp [ns]", "suffix"]
            )

    # Drop helper columns
    for col in ["frame_name", "frame_path"]:
        if col in filtered_df.columns:
            filtered_df.drop(columns=[col], inplace=True)

    filtered_df.to_csv(augmented_csv_path, index=False)
    print(f"\t\tOK Pruning {rec_dict.get('recording_id')} frame. Got {len(filtered_df)} frames.")

    return augmented_csv_path
