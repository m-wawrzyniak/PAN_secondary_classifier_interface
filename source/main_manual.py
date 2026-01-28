from pathlib import Path
from pprint import pprint

import a01_frames_extraction as a01
import a02_classify as a02
import a03_postprocessing as a03

from source import config as conf

REC_DIR = Path("/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Timeseries Data + Scene Video/2024-02-29_13-07-05-b2a83ebc")

SECTIONS_PATH = Path("/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative/sections.csv")
FACE_DETECTIONS = Path("/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/Sit&Face_FACE-MAPPER_Faces_Manipulative/face_detections.csv")

MODEL_PATH = Path("/home/mateusz-wawrzyniak/PycharmProjects/PAN_secondary_classifier_interface/model/sec_model.pth")
OUTPUT_DIR = Path("/media/mateusz-wawrzyniak/Extreme SSD/IP_PAN/interface_test")

run_range = (1, 3)

# a01 - preprocessing
rec_context = a01.prepare_recording_for_inference(
    recording_dir=REC_DIR,
    output_root=OUTPUT_DIR
)

sections = a01.get_recording_sections(
    sections_csv_path=SECTIONS_PATH,
    recording_id=rec_context["recording_id"]
)

rec_context["sections"] = sections
pprint(rec_context)

if run_range[0] <= 1 <= run_range[1]:
    a01.extract_frames(
        rec_dict=rec_context,
        mapper_detections=FACE_DETECTIONS
    )

# a02 - classification
if run_range[0] <= 2 <= run_range[1]:
    df = a02.classify_recording(
        rec_dict=rec_context,
        model_path=MODEL_PATH,
        input_size=conf.CNN_INPUT_SIZE,
        dec_threshold=conf.OPT_PROB_THRESHOLD)

# a03 - postprocessing
if run_range[0] <= 3 <= run_range[1]:
    a03.prune_face_frames_local(
        rec_dict=rec_context,
        aggregate_csv_path=FACE_DETECTIONS)
