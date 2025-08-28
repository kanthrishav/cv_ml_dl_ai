#!/usr/bin/env python3
"""
train_logo_detector.py

1. Downloads the FlickrLogos-47 dataset (re-annotated for detection).
2. Converts PascalVOC XML annotations to YOLOv5 format.
3. Trains a YOLOv5-n detector on the 47 classes.
4. Exports the best model to TorchScript (brand_detector_ts.pt).
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path

# 1st-party dependencies: ultralytics (YOLOv5), xmltodict
# Install via: pip install ultralytics xmltodict tqdm
from ultralytics import YOLO
import xmltodict
from tqdm import tqdm

# 1. Constants & URLs
BASE_DIR      = Path(__file__).parent.resolve()
# updated to point at the correct repo and branch:
DATA_ZIP_URL  = "http://123.57.42.89/Dataset_ict/LogoDet-3K.zip"
DATA_ZIP_FILE = BASE_DIR / "LogoDet-3K.zip"
RAW_DATA_DIR  = BASE_DIR / "LogoDet-3K"
IMAGES_DIR    = RAW_DATA_DIR / "images"
ANNOT_DIR     = RAW_DATA_DIR / "annotations"
YOLO_DATA_DIR = BASE_DIR / "yolo_data"
MODEL_OUT     = BASE_DIR / "runs/train/logo_detect/weights/best.pt"
TS_MODEL_OUT  = BASE_DIR / "brand_detector_ts.pt"
NUM_CLASSES   = 47  # FlickrLogos-47

def download_and_extract():
    if not RAW_DATA_DIR.exists():
        print("[1/5] Downloading dataset ZIP...")
        urllib.request.urlretrieve(DATA_ZIP_URL, DATA_ZIP_FILE)
        print("[2/5] Extracting...")
        with zipfile.ZipFile(DATA_ZIP_FILE, "r") as z:
            z.extractall(BASE_DIR)
        DATA_ZIP_FILE.unlink()
    else:
        print("Dataset already downloaded.")

def convert_annotations():
    """
    Converts the XML annotations in ANNOT_DIR into YOLOv5 text files under yolo_data/images/{train,val}/labels.
    """
    print("[3/5] Converting annotations to YOLO format...")
    # FlickrLogos-47 comes with train/val split files
    # assumed structure: annotations/train/*.xml and annotations/val/*.xml
    for split in ("train", "val"):
        img_split = YOLO_DATA_DIR / "images" / split
        lbl_split = YOLO_DATA_DIR / "labels" / split
        img_split.mkdir(parents=True, exist_ok=True)
        lbl_split.mkdir(parents=True, exist_ok=True)

        xml_dir = ANNOT_DIR / split
        for xml_file in tqdm(list(xml_dir.glob("*.xml")), desc=f"Converting {split}"):
            # parse XML
            with open(xml_file) as f:
                doc = xmltodict.parse(f.read())
            objs = doc["annotation"]["object"]
            if not isinstance(objs, list):
                objs = [objs]
            # copy image
            img_name = doc["annotation"]["filename"]
            src_img  = IMAGES_DIR / split / img_name
            dst_img  = img_split / img_name
            if not dst_img.exists():
                shutil.copy(src_img, dst_img)
            # write label file
            h = int(doc["annotation"]["size"]["height"])
            w = int(doc["annotation"]["size"]["width"])
            yolo_lines = []
            for o in objs:
                cls = int(o["name"])  # classes are 0-indexed in this dataset
                bb  = o["bndbox"]
                xmin, ymin = int(bb["xmin"]), int(bb["ymin"])
                xmax, ymax = int(bb["xmax"]), int(bb["ymax"])
                # convert to YOLO format
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
            with open(lbl_split / (xml_file.stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

def write_data_yaml():
    """
    Writes the data config for YOLOv5.
    """
    print("[4/5] Writing data.yaml...")
    content = f"""
train: {YOLO_DATA_DIR}/images/train
val:   {YOLO_DATA_DIR}/images/val

nc: {NUM_CLASSES}
names: [{','.join(f"'{i}'" for i in range(NUM_CLASSES))}]
"""
    (YOLO_DATA_DIR / "data.yaml").write_text(content.strip())

def train_yolo():
    """
    Uses Ultralytics YOLOv5 API to train a nano model.
    """
    print("[5/5] Starting training...")
    model = YOLO("yolov5n.pt")  # pre-trained nano
    model.train(
        data=str(YOLO_DATA_DIR / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=5,
        project=str(BASE_DIR / "runs" / "train"),
        name="logo_detect"
    )
    print("Training complete. Best model saved to:", MODEL_OUT)

def export_torchscript():
    """
    Converts the best PyTorch checkpoint to TorchScript for lean inference.
    """
    print("Exporting to TorchScript...")
    model = YOLO(MODEL_OUT).model
    example = (torch.zeros(1,3,640,640).float())
    ts = torch.jit.trace(model, example)
    ts.save(str(TS_MODEL_OUT))
    print("TorchScript model saved to:", TS_MODEL_OUT)

if __name__ == "__main__":
    download_and_extract()
    convert_annotations()
    write_data_yaml()
    train_yolo()
    export_torchscript()
    print("** All done! ** Run 'detect_logos_pi.py' on your Pi to inference. ")
