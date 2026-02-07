"""
Batch prediction: run model on a folder of images, save annotated images and a result markdown.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO
import supervision as sv
import numpy as np

from utils import LABEL_MAPPING


def _display_name(original_class: str) -> str:
    """Map original class to display name (e.g. 4索 -> 4条)."""
    name = LABEL_MAPPING.get(original_class, original_class)
    return name.replace("索", "条") if "索" in name else name


def _default_font_path() -> str:
    """Return a font path that supports Chinese on current OS."""
    if sys.platform == "win32":
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def run_predict_batch(
    pics_dir: Path,
    result_dir: Path,
    model_path: Path,
    conf_threshold: float = 0.3,
    font_path: str = "",
) -> None:
    """
    Run prediction on all images in pics_dir, save annotated images and 结果.md to result_dir.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    result_dir.mkdir(parents=True, exist_ok=True)

    font_path = font_path or _default_font_path()
    model = YOLO(str(model_path))
    box_annotator = sv.BoxAnnotator()
    if font_path:
        label_annotator = sv.RichLabelAnnotator(
            font_path=font_path,
            font_size=24,
            smart_position=True,
        )
    else:
        label_annotator = sv.LabelAnnotator()

    image_paths = sorted(pics_dir.glob("*.jpg")) + sorted(pics_dir.glob("*.jpeg")) + sorted(pics_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {pics_dir}")

    all_rows = []
    global_index = 0

    for img_path in image_paths:
        try:
            image = Image.open(str(img_path)).convert("RGB")
            # Apply EXIF orientation so landscape photos stay landscape (many phones store them rotated)
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            raise RuntimeError(f"Failed to open image {img_path}: {e}") from e

        img_w, img_h = image.size
        results = model(image, conf=conf_threshold, imgsz=1024, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > 0.5]

        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            original = model.model.names[int(class_id)]
            disp = _display_name(original)
            labels.append(f"{disp} {confidence:.2f}")

        annotated = box_annotator.annotate(scene=np.array(image), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        out_name = f"{img_path.stem}_result{img_path.suffix}"
        out_path = result_dir / out_name
        Image.fromarray(annotated).save(str(out_path))
        print(f"Saved {out_path}")

        xyxy = detections.xyxy
        for i in range(len(detections)):
            global_index += 1
            x1, y1, x2, y2 = xyxy[i]
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            cx_pct = (cx_px / img_w) * 100.0
            cy_pct = (cy_px / img_h) * 100.0
            w_px = x2 - x1
            h_px = y2 - y1
            class_id = int(detections.class_id[i])
            original = model.model.names[class_id]
            mapped = _display_name(original)
            all_rows.append({
                "index": global_index,
                "image": img_path.name,
                "cx_px": cx_px,
                "cy_px": cy_px,
                "cx_pct": cx_pct,
                "cy_pct": cy_pct,
                "w_px": w_px,
                "h_px": h_px,
                "original": original,
                "mapped": mapped,
            })

    _write_result_md(result_dir / "结果.md", all_rows)
    print(f"Wrote {result_dir / '结果.md'}")


def _write_result_md(path: Path, rows: list) -> None:
    """Write detection list to 结果.md."""
    lines = [
        "# 检测结果",
        "",
        "| 序号 | 图片 | 中心点(像素) | 中心点(%) | 宽度(像素) | 高度(像素) | 原始分类 | 映射后真实分类 |",
        "|------|------|--------------|-----------|------------|------------|----------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['image']} | ({r['cx_px']:.1f}, {r['cy_px']:.1f}) | ({r['cx_pct']:.2f}%, {r['cy_pct']:.2f}%) | {r['w_px']:.1f} | {r['h_px']:.1f} | {r['original']} | {r['mapped']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict images and write result images + 结果.md")
    parser.add_argument("--pics_dir", type=str, default="test/1/pics", help="Folder containing input images")
    parser.add_argument("--result_dir", type=str, default="test/1/result", help="Folder for result images and 结果.md")
    parser.add_argument("--model", type=str, default="runs/detect/train2/weights/best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--font_path", type=str, default="", help="Font path for labels (optional)")
    args = parser.parse_args()

    run_predict_batch(
        pics_dir=Path(args.pics_dir),
        result_dir=Path(args.result_dir),
        model_path=Path(args.model),
        conf_threshold=args.conf,
        font_path=args.font_path or _default_font_path(),
    )


if __name__ == "__main__":
    main()
