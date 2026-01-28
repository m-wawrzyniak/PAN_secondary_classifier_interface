import pandas as pd
from pathlib import Path

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
