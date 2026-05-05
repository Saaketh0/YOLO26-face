# Converts the WIDER FACE dataset from a modified COCO format to YOLO
# Stores all files in the wider_face_yolo folder
# Took around 5 minutes to process all of this locally on a Macbook Air M3

from datasets import load_dataset, Image as HFImage
from pathlib import Path
from PIL import Image as PILImage
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import shutil

DATASET_NAME = "CUHK-CSE/wider_face"
OUTPUT_DIR = Path("wider_face_yolo")
NUM_WORKERS = min(8, os.cpu_count() or 8)

def link_or_copy(src_path: str, dst_path: Path) -> None:
    if dst_path.exists():
        return
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

def make_yolo_lines(sample: dict, img_w: int, img_h: int) -> list[str]:
    lines = []
    bboxes = sample["faces"]["bbox"]
    invalid_flags = sample["faces"]["invalid"]

    for bbox, invalid in zip(bboxes, invalid_flags):
        if invalid:
            continue

        x_min, y_min, box_w, box_h = bbox
        if box_w <= 0 or box_h <= 0:
            continue

        x1 = max(0, min(x_min, img_w))
        y1 = max(0, min(y_min, img_h))
        x2 = max(0, min(x_min + box_w, img_w))
        y2 = max(0, min(y_min + box_h, img_h))

        clipped_w = x2 - x1
        clipped_h = y2 - y1

        if clipped_w <= 0 or clipped_h <= 0:
            continue

        x_center = x1 + clipped_w / 2
        y_center = y1 + clipped_h / 2

        lines.append(
            f"0 {x_center / img_w:.6f} {y_center / img_h:.6f} {clipped_w / img_w:.6f} {clipped_h / img_h:.6f}"
        )

    return lines

def process_sample(sample: dict, i: int, yolo_split: str, images_dir: Path, labels_dir: Path) -> None:
    image_info = sample["image"]
    src_path = image_info.get("path")
    image_bytes = image_info.get("bytes")

    image_name = f"{yolo_split}_{i:06d}.jpg"
    label_name = f"{yolo_split}_{i:06d}.txt"

    image_path = images_dir / yolo_split / image_name
    label_path = labels_dir / yolo_split / label_name

    if src_path is not None:
        with PILImage.open(src_path) as img:
            img_w, img_h = img.size
        link_or_copy(src_path, image_path)
    else:
        with PILImage.open(BytesIO(image_bytes)) as img:
            img_w, img_h = img.size
        if not image_path.exists():
            image_path.write_bytes(image_bytes)

    yolo_lines = make_yolo_lines(sample, img_w, img_h)
    label_path.write_text("\n".join(yolo_lines))

def write_data_yaml(output_dir: Path) -> None:
    data_yaml = "path: wider_face_yolo\ntrain: images/train\nval: images/val\n\nnames:\n  0: face\n"
    (output_dir / "data.yaml").write_text(data_yaml)

def main() -> None:
    ds = load_dataset(DATASET_NAME, trust_remote_code=True)
    ds = ds.cast_column("image", HFImage(decode=False))

    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"

    split_map = {
        "train": "train",
        "validation": "val",
    }

    for hf_split, yolo_split in split_map.items():
        (images_dir / yolo_split).mkdir(parents=True, exist_ok=True)
        (labels_dir / yolo_split).mkdir(parents=True, exist_ok=True)

        split_ds = ds[hf_split]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_sample,
                    sample,
                    i,
                    yolo_split,
                    images_dir,
                    labels_dir,
                )
                for i, sample in enumerate(split_ds)
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(split_ds),
                desc=f"Converting {hf_split} -> {yolo_split}",
            ):
                future.result()

    write_data_yaml(OUTPUT_DIR)

if __name__ == "__main__":
    main()