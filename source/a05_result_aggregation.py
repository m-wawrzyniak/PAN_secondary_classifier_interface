import pandas as pd
from pathlib import Path
import numpy as np

def aggregate_all_augmented(recording_contexts: list, output_root: Path):
    """Aggregate all per-recording augmented_face_frames.csv into one CSV"""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregated_csv = output_root / "augmented_face_detections.csv"

    all_dfs = []
    for rec_ctx in recording_contexts:
        aug_csv = Path(rec_ctx["output_dir"]) / "augmented_face_frames.csv"
        if not aug_csv.exists():
            print(f"! Missing augmented CSV for {rec_ctx['recording_id']}, skipping")
            continue
        df = pd.read_csv(aug_csv)
        all_dfs.append(df)

    if not all_dfs:
        print("\t\t! No frames to aggregate. Exiting.")
        return

    aggregated_df = pd.concat(all_dfs, ignore_index=True)
    aggregated_df.to_csv(aggregated_csv, index=False)
    print(f"\tOK Aggregated CSV saved. Got {len(aggregated_df)} frames.")


def prune_fixations_on_faces_aggregated(
    fixations_csv: Path,
    augmented_agg_csv: Path,
    tolerance_ms: float = 25.0,
    save_path: Path | None = None,
):
    """
    Keep only fixations that overlap with face frames (± tolerance).

    A fixation is kept if at least one face frame timestamp falls within:
        [start_ts - tol, end_ts + tol]

    Works across all recordings.
    """

    fix_df = pd.read_csv(fixations_csv)
    frames_df = pd.read_csv(augmented_agg_csv)

    tol_ns = tolerance_ms * 1e6

    # recording_id -> sorted numpy array of frame timestamps
    frames_by_rec = {
        rec_id: np.sort(group["timestamp [ns]"].to_numpy(dtype=float))
        for rec_id, group in frames_df.groupby("recording id")
    }

    keep_mask = []

    for _, row in fix_df.iterrows():
        rec_id = row["recording id"]

        if rec_id not in frames_by_rec:
            keep_mask.append(False)
            continue

        ts_frames = frames_by_rec[rec_id]

        start = row["start timestamp [ns]"] - tol_ns
        end = row["end timestamp [ns]"] + tol_ns

        # Find first frame >= start
        idx = np.searchsorted(ts_frames, start, side="left")

        keep = (idx < len(ts_frames)) and (ts_frames[idx] <= end)
        keep_mask.append(keep)

    pruned_df = fix_df[np.array(keep_mask)].copy()

    if save_path is None:
        save_path = fixations_csv

    pruned_df.to_csv(save_path, index=False)
    print(
        f"\tOK Pruned fixations_on_faces (aggregated, ±{tolerance_ms} ms) "
        f"({len(pruned_df)}/{len(fix_df)} rows)"
    )

    return pruned_df



def prune_gaze_on_face_aggregated(
        gaze_csv: Path,
        augmented_agg_csv: Path,
        save_path: Path | None = None,
):
    """
    Keep only gaze timestamps that fall within any of the augmented face frames.
    Works across all recordings, robust to different sampling rates.

    Parameters
    ----------
    gaze_csv : Path
        Original gaze_on_face.csv
    augmented_agg_csv : Path
        Aggregated augmented_face_frames.csv
    save_path : Path | None
        Where to save the pruned gaze CSV. If None, overwrites original.
    """
    gaze_df = pd.read_csv(gaze_csv)
    frames_df = pd.read_csv(augmented_agg_csv)

    # Compute frame period per recording (approx from sorted timestamps)
    frames_df["timestamp [ns]"] = frames_df["timestamp [ns]"].astype(float)
    frames_df = frames_df.sort_values(["recording id", "timestamp [ns]"])

    # Build frame intervals: start = timestamp - half_frame, end = timestamp + half_frame
    intervals_by_rec = {}
    for rec_id, rec_frames in frames_df.groupby("recording id"):
        ts = rec_frames["timestamp [ns]"].values
        if len(ts) < 2:
            half_frame = 25_000_000  # fallback: 20 Hz → 50ms → half = 25ms
        else:
            # approximate half frame duration from median difference
            median_diff = pd.Series(ts[1:] - ts[:-1]).median()
            half_frame = median_diff / 2
        # Store intervals as list of (start, end)
        intervals = [(t - half_frame, t + half_frame) for t in ts]
        intervals_by_rec[rec_id] = intervals

    def is_in_any_interval(row):
        rec_id = row["recording id"]
        t = float(row["timestamp [ns]"])
        if rec_id not in intervals_by_rec:
            return False
        intervals = intervals_by_rec[rec_id]
        # Binary search can be used for large datasets, but pandas apply is okay for medium
        # For millions of rows, consider numpy searchsorted
        # We'll do numpy vectorized check
        starts = pd.Series([s for s, e in intervals])
        ends = pd.Series([e for s, e in intervals])
        mask = (t >= starts) & (t <= ends)
        return mask.any()

    # Apply check
    keep_mask = gaze_df.apply(is_in_any_interval, axis=1)
    pruned_df = gaze_df[keep_mask].copy()

    if save_path is None:
        save_path = gaze_csv

    pruned_df.to_csv(save_path, index=False)
    print(f"\tOK Pruned gaze_on_face (aggregated, ±frame) ({len(pruned_df)}/{len(gaze_df)} rows)")
    return pruned_df

