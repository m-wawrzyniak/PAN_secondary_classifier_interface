# PAN Secondary Face Classifier — User Interface

This project provides a **lightweight interface** for running the secondary face classification model on your recordings. It is **designed for researchers** who want to classify their recordings **without training the model** or handling the full pipeline.

---

## Overview

The pipeline performs the following steps for each recording:

1. **Frame extraction**: Crops faces from the original video using precomputed face detections.
2. **Classification**: Runs the pre-trained CNN to classify each extracted face as `is_face=1` or `0`.
3. **Postprocessing**:
   - Smooths the classification along time (optional, window-based).
   - Generates per-recording `augmented_face_frames.csv` with valid frames.
4. **Optional HTML visualization**: Creates a paginated HTML for quick inspection of faces / non-faces.

After processing all recordings, you can **aggregate augmented results** and optionally **prune gaze/fixation CSVs**.

---

## Requirements

- Python ≥ 3.10  
- OS: Linux / Mac / Windows  
- Install required packages:

```bash
pip install -r requirements.txt
```

**Key libraries**: `torch`, `torchvision`, `opencv-python`, `pandas`, `PIL`, `tqdm`

**Optional**: Activate the project `.venv` before running:

```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

---

## Project Structure

```
project_root/
│
├─ model/
│   └─ sec_model.pth            # Pre-trained CNN model
├─ interface/                   # Interface scripts
│   ├─ a01_frames_extraction.py
│   ├─ a02_classify.py
│   ├─ a03_postprocessing.py
│   └─ run_interface.py         # Main pipeline entry
├─ data/
│   ├─ face_detections.csv      # Face detection metadata
│   └─ sections.csv             # Sections info (start/end timestamps)
└─ recordings/
    ├─ 2024-02-29_13-07-05/
    └─ 2024-03-01_14-22-10/
```

- **recordings/**: Folder containing all recordings to process (each recording in a separate folder).
- **data/**: Shared folder for `face_detections.csv` and `sections.csv`.
- **model/sec_model.pth**: Pre-trained classifier (automatically loaded from project root).

---

## Usage

Run the interface pipeline from the **project root**:

```bash
python -m interface.run_interface \
    --rec_dir_root "/path/to/recordings" \
    --output_root "/path/to/output" \
    --data_root "/path/to/data" \
    [--run_range 1 3] \
    [--aggregate] \
    [--smooth_window 3] \
    [--html]
```

### Arguments

| Argument              | Description |
|-----------------------|-------------|
| `--rec_dir_root`      | Folder containing recordings to process. Each recording in a separate folder. |
| `--output_root`       | Where all extracted frames, CSVs, and results will be saved. |
| `--data_root`         | Folder containing `sections.csv` and `face_detections.csv`. |
| `--run_range`         | Two integers `[start end]` indicating which stages to run: 1=extract, 2=classify, 3=postprocess. Default `[1 3]`. |
| `--aggregate`         | Aggregate all per-recording augmented frames into a single CSV after processing. |
| `--smooth_window`     | Optional smoothing window (odd integer, e.g., `3`). Set `0` to disable smoothing. |
| `--html`              | Generate simple paginated HTML visualization for quick inspection. |

---

### Examples

**Process all recordings and generate aggregated CSVs**:

```bash
python -m interface.run_interface \
    --rec_dir_root "/media/recordings" \
    --output_root "/media/output" \
    --data_root "/media/data" \
    --aggregate
```

**Process a single recording, with 3-frame smoothing and HTML visualization**:

```bash
python -m interface.run_interface \
    --rec_dir_root "/media/recordings/2024-02-29_13-07-05" \
    --output_root "/media/output" \
    --data_root "/media/data" \
    --run_range 1 3 \
    --smooth_window 3 \
    --html
```

---

## Output

For each recording:

- `extracted/` — Cropped face frames used for classification.
- `model_class.csv` — Initial per-frame predictions (`is_face=0/1`).
- `augmented_face_frames.csv` — Postprocessed, validated frames.
- `result_html/` — Optional HTML pages for visual inspection.

After aggregation:

- `aggregated/augmented_face_detections.csv` — Combined frames from all recordings.
- Optional pruned versions of:
  - `fixations_on_faces.csv`
  - `gaze_on_face.csv`

---

## Notes / Tips

- Make sure **face detections are aligned with sections.csv** timestamps.
- Use `--smooth_window > 0` if single-frame misclassifications are likely.
- The HTML visualization is optional but very useful for verifying classification quality quickly.
- Aggregation should be run **after all recordings** have been processed to generate a complete dataset for downstream analysis.

---

## Contact / Support

If you encounter issues, check:

- Python version
- Installed dependencies (`pip list`)
- Video files accessibility

For questions about the pipeline or interpretation, reach out to the project maintainer.
