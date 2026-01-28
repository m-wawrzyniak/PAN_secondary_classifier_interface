import json
from pathlib import Path
import cv2
import pandas as pd

import config as conf


def prepare_recording_for_inference(
    recording_dir: str | Path,
    output_root: str | Path,
    strict: bool = False,
) -> dict:
    """
    Prepare a single recording for inference.

    Validates recording structure, loads metadata, and creates
    an output directory for downstream inference steps.

    Parameters
    ----------
    recording_dir : str or Path
        Path to a single recording directory.
    output_root : str or Path
        Base directory where inference outputs will be written.
    strict : bool
        If True, raises errors on validation issues.
        If False, collects warnings and continues where possible.

    Returns
    -------
    dict
        Recording context dictionary.
    """

    recording_dir = Path(recording_dir)
    output_root = Path(output_root)

    if not recording_dir.exists() or not recording_dir.is_dir():
        raise ValueError(f"\t\tRecording dir does not exist: {recording_dir}")

    warnings = []

    # --- Required files ---
    info_path = recording_dir / "info.json"
    timestamps_path = recording_dir / "world_timestamps.csv"
    mp4_files = list(recording_dir.glob("*.mp4"))

    def _handle_issue(msg: str):
        if strict:
            raise RuntimeError(msg)
        warnings.append(msg)

    if not info_path.exists():
        _handle_issue("\t\tMissing info.json")

    if not timestamps_path.exists():
        _handle_issue("\t\tMissing world_timestamps.csv")

    if len(mp4_files) != 1:
        _handle_issue(f"\t\tExpected 1 mp4 file, found {len(mp4_files)}")

    # Bail early if strict and fatal
    if strict and warnings:
        raise RuntimeError("\t\tRecording validation failed")

    video_path = mp4_files[0] if mp4_files else None

    # --- Load info.json ---
    recording_id = None
    recording_name = None
    start_time = None
    gaze_frequency = None

    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        recording_id = info.get("recording_id")
        recording_name = info.get("template_data", {}).get("recording_name")
        start_time = info.get("start_time")
        gaze_frequency = info.get("gaze_frequency")

        if gaze_frequency != 200:
            _handle_issue(f"\t\tUnexpected gaze_frequency={gaze_frequency} (expected 200)")

    # --- Output directory ---
    output_dir = output_root / recording_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build context ---
    context = {
        "recording_id": recording_id,
        "recording_name": recording_name,
        "recording_dir": str(recording_dir),
        "video_path": str(video_path) if video_path else None,
        "timestamps_path": str(timestamps_path) if timestamps_path.exists() else None,
        "output_dir": str(output_dir),
        "start_time": start_time,
        "gaze_frequency": gaze_frequency,
        "warnings": warnings,
    }

    return context


def get_recording_sections(
    sections_csv_path: str | Path,
    recording_id: str,
    strict: bool = False,
) -> list[dict]:
    """
    Fetch all section time ranges for a given recording_id.

    Parameters
    ----------
    sections_csv_path : str or Path
        Path to sections.csv
    recording_id : str
        Recording ID to filter by
    strict : bool
        If True, raise error when no sections are found

    Returns
    -------
    list of dict
        Each dict contains section_id, start_time_ns, end_time_ns
    """

    sections_csv_path = Path(sections_csv_path)
    if not sections_csv_path.exists():
        raise FileNotFoundError(f"\t\tsections.csv not found: {sections_csv_path}")

    df = pd.read_csv(sections_csv_path)

    rows = df[df["recording id"] == recording_id]

    if rows.empty:
        if strict:
            raise RuntimeError(f"\t\tNo sections found for recording_id={recording_id}")
        return []

    sections = []
    for _, row in rows.iterrows():
        sections.append({
            "section_id": row.get("section id"),
            "start_time_ns": int(row["section start time [ns]"]),
            "end_time_ns": int(row["section end time [ns]"]),
            "start_event": row.get("start event name"),
            "end_event": row.get("end event name"),
        })

    return sections


def _load_and_filter_csv(mapper_detections: str, recording_id: str):
    """
    Load face_detections.csv and filter by recording_id.
    """
    df = pd.read_csv(mapper_detections)
    df = df[df["recording id"] == recording_id].reset_index(drop=True)
    return df


def extract_frames(rec_dict: dict, mapper_detections: Path|str):
    recording_id = rec_dict["recording_id"]
    video_path = rec_dict["video_path"]
    extract_dir = Path(rec_dict["output_dir"])
    extract_dir.mkdir(parents=True, exist_ok=True)

    df = _load_and_filter_csv(mapper_detections, recording_id)
    if df.empty:
        print(f"\t\t! No face detections for {recording_id}")
        return

    df = df.dropna(subset=["p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]"]).reset_index(drop=True)
    df = df.sort_values("timestamp [ns]").reset_index(drop=True)

    df["suffix"] = 0
    last_ts = None
    counter = -1
    for i, row in df.iterrows():
        ts = int(row["timestamp [ns]"])
        if ts != last_ts:
            counter = 0
            last_ts = ts
        else:
            counter += 1
        df.at[i, "suffix"] = counter

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"\t\t! Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rec_start_ns = int(rec_dict.get("start_time", 0))
    sections = rec_dict.get("sections", [])

    total = len(df)
    for i, row in df.iterrows():
        ts = int(row["timestamp [ns]"])
        suffix = int(row["suffix"])
        time_offset = (ts - rec_start_ns) / 1e9
        if time_offset < 0:
            continue
        frame_idx = int(round(time_offset * fps))
        if frame_idx < 0 or frame_idx >= frame_count:
            continue

        if sections:
            if not any(section["start_time_ns"] <= ts <= section["end_time_ns"] for section in sections):
                continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        p1x, p1y, p2x, p2y = map(int, [row["p1 x [px]"], row["p1 y [px]"], row["p2 x [px]"], row["p2 y [px]"]])
        h, w = frame.shape[:2]
        box_w, box_h = p2x - p1x, p2y - p1y
        pad = int(max(box_w, box_h) * conf.IMAGE_PAD_RATIO)

        x1 = max(0, p1x - pad)
        y1 = max(0, p1y - pad)
        x2 = min(w, p2x + pad)
        y2 = min(h, p2y + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        cropped = frame[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        scale = conf.CNN_INPUT_SIZE / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = (conf.CNN_INPUT_SIZE - new_h) // 2
        bottom = conf.CNN_INPUT_SIZE - new_h - top
        left = (conf.CNN_INPUT_SIZE - new_w) // 2
        right = conf.CNN_INPUT_SIZE - new_w - left
        squared = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

        filename = extract_dir / f"{ts}_{suffix}.jpg"
        cv2.imwrite(str(filename), squared)
        df.at[i, "frame_path"] = str(filename)

        if (i + 1) % 300 == 0 or (i + 1) == total:
            print(f"\t\tProgress: {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")

    csv_out_path = extract_dir / "face_frames.csv"
    df.to_csv(csv_out_path, index=False)
    cap.release()
    print(f"\t\tOK Saved {total} face frames for {recording_id}")
    return str(csv_out_path)
