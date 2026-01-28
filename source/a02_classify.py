import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from FaceVerifierCNN import FaceVerifierCNN

def load_model(model_path, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceVerifierCNN(input_size=input_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"\t\tmodel_state_dict missing in checkpoint! Keys: {checkpoint.keys()}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

def classify_frame(model, device, img_path, transform, dec_threshold):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        prob = torch.sigmoid(logits).item()
    return 1 if prob >= dec_threshold else 0

def classify_recording(rec_dict, model_path, input_size, dec_threshold):
    model, device = load_model(model_path, input_size)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    extraction_dir = Path(rec_dict["output_dir"])
    jpg_files = sorted(p for p in extraction_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
    if not jpg_files:
        print(f"\t\tNo extracted frames found in {extraction_dir} — skipping")
        return None

    sections = rec_dict.get("sections", [])
    results = []
    desc_str = f"Classifying {rec_dict['recording_id']}"
    for frame_path in tqdm(jpg_files, desc=f"{desc_str:<30}", unit="frame"):
        ts = int(frame_path.stem.split("_")[0])
        if sections:
            if not any(section["start_time_ns"] <= ts <= section["end_time_ns"] for section in sections):
                continue
        pred = classify_frame(model, device, frame_path, transform, dec_threshold)
        results.append((frame_path.name, pred))

    out_csv = extraction_dir / "model_class.csv"
    df = pd.DataFrame(results, columns=["frame", "is_face"])
    df.to_csv(out_csv, index=False)
    print(f"\t\tOK Saved predictions")
    return df
