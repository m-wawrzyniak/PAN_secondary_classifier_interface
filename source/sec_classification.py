#!/usr/bin/env python3
from pathlib import Path
import argparse

import a01_frames_extraction as a01
import a02_classify as a02
import a03_postprocessing as a03
import a04_html_visuals as a04
import a05_result_aggregation as a05

import config as conf

"""
source .venv/bin/activate

python -u source/sec_classification.py \
  --rec_dir_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Timeseries Data + Scene Video" \
  --output_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/interface_final" \
  --data_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative" \
  --run_range 3 3 \
  --aggregate \
  --html \
  > run.log 2>&1

tail -f run.log

python -u source/sec_classification.py \
  --rec_dir_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/video_test" \
  --output_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/interface_smoothed" \
  --data_root "/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative" \
  --run_range 1 3 \
  --aggregate \
  --html \
  > run.log 2>&1
"""

def process_recording(rec_dir: Path, output_root: Path, sections_csv: Path, face_detections: Path, model_path: Path, run_range=(1, 3)):
    rec_context = a01.prepare_recording_for_inference(recording_dir=rec_dir, output_root=output_root)
    sections = a01.get_recording_sections(sections_csv_path=sections_csv, recording_id=rec_context["recording_id"])
    rec_context["sections"] = sections

    # Stage 1: extract frames
    if run_range[0] <= 1 <= run_range[1]:
        print(f"\ta01. Frame extraction...")
        a01.extract_frames(rec_dict=rec_context, mapper_detections=face_detections)

    # Stage 2: classify frames
    if run_range[0] <= 2 <= run_range[1]:
        print(f"\ta02. Classification...")
        a02.classify_recording(
            rec_dict=rec_context,
            model_path=model_path,
            input_size=conf.CNN_INPUT_SIZE,
            dec_threshold=conf.OPT_PROB_THRESHOLD
        )

    # Stage 3: prune / postprocess
    if run_range[0] <= 3 <= run_range[1]:
        if run_range[0] <= 3 <= run_range[1]:
            model_csv = Path(rec_context["output_dir"], "model_class.csv")

            if not model_csv.exists():
                print("\t\ta03. Skipping postprocessing (no model_class.csv)")
            else:
                print(f"\ta03. Postprocessing...")

                a03.smooth_classification(
                    csv_path=model_csv,
                    window=conf.SMOOTH_WINDOW
                )

                a03.prune_face_frames_local(
                    rec_dict=rec_context,
                    aggregate_csv_path=face_detections
                )

    return rec_context

def main():
    parser = argparse.ArgumentParser(description="PAN secondary classifier: single-recording or batch inference")
    parser.add_argument("--rec_dir_root", type=str, required=True, help="Folder containing recordings")
    parser.add_argument("--output_root", type=str, required=True, help="Output folder for frames, CSVs, etc.")
    parser.add_argument("--data_root", type=str, required=True, help="Folder containing sections.csv and face_detections.csv")
    parser.add_argument("--run_range", type=int, nargs=2, default=[1, 3], help="Which stages to run: 1=extract,2=classify,3=postprocess")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all augmented frames at the end")
    parser.add_argument("--html", action="store_true", help="Generate simple HTML validation output per recording"
    )
    args = parser.parse_args()

    rec_dir_root = Path(args.rec_dir_root)
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)

    model_path = Path.cwd() / "model" / "sec_model.pth"
    sections_csv = data_root / "sections.csv"
    face_detections = data_root / "face_detections.csv"
    fixations_path = data_root / "fixations_on_face.csv"
    gaze_path = data_root / "gaze_on_face.csv"

    recording_dirs = [p for p in rec_dir_root.iterdir() if p.is_dir()]
    recording_contexts = []
    dir_cnt = len(recording_dirs)
    cur_cnt = 0

    print(f"\nRUNNING SECONDARY FACE CLASSIFICATION FOR DIRECTORY: {rec_dir_root}")

    for rec_dir in recording_dirs:
        cur_cnt += 1
        print(f"\n\tProcessing recording: {rec_dir.name} ({cur_cnt}/{dir_cnt})")
        rec_ctx = process_recording(
            rec_dir=rec_dir,
            output_root=output_root,
            sections_csv=sections_csv,
            face_detections=face_detections,
            model_path=model_path,
            run_range=tuple(args.run_range)
        )
        recording_contexts.append(rec_ctx)

        if args.html:
            print(f"\ta04. HTML visualization...")
            # CSV with classification results
            model_csv = Path(rec_ctx["output_dir"]) / "model_class.csv"
            if model_csv.exists():
                a04.export_html_paginated(
                    name=rec_ctx["recording_id"],
                    csv_path=model_csv,
                    extracted_path=Path(rec_ctx["output_dir"])
                )
            else:
                print(f"\t\t! model_class.csv missing for {rec_ctx['recording_id']}, skipping HTML export")

    if args.aggregate:
        print(f"\ta05. Aggregating results...")
        a05.aggregate_all_augmented(
            recording_contexts=recording_contexts,
            output_root=output_root)

        a05.prune_fixations_on_faces_aggregated(
            fixations_csv=fixations_path,
            augmented_agg_csv= Path(output_root, "augmented_face_detections.csv"),
            save_path=Path(output_root,"augmented_fixations_on_face.csv")
        )

        a05.prune_gaze_on_face_aggregated(
            gaze_csv=gaze_path,
            augmented_agg_csv=Path(output_root, "augmented_face_detections.csv"),
            save_path=Path(output_root,"augmented_gaze_on_face.csv")
        )

    print(f"\nSECONDARY CLASSIFICATION DONE.")

if __name__ == "__main__":
    main()
