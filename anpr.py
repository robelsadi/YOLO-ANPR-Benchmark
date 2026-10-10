"""YOLO + EasyOCR automatic number-plate recognition."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Optional, Union
import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO

PathLike = Union[str, int, Path]
OFFICIAL_MODELS = {"yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"}

class ANPR:
    def __init__(self, model_path: str, device: str = "auto", confidence: float = 0.25,
                 languages: Optional[list[str]] = None) -> None:
        model_file = Path(model_path)
        if not model_file.exists() and model_path not in OFFICIAL_MODELS:
            raise FileNotFoundError(
                f"Model file not found: {model_path}\nPut your plate-trained YOLO weights at "
                "models/number-plate.pt and pass --model models/number-plate.pt."
            )
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        if device == "0" and not torch.cuda.is_available():
            print("WARNING: CUDA is unavailable; switching to CPU.", file=sys.stderr)
            device = "cpu"
        self.device, self.confidence = device, confidence
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(languages or ["en"], gpu=(device != "cpu"))
        self.names = self.model.names
        if model_path in OFFICIAL_MODELS:
            print("WARNING: Official YOLO26 weights are general-purpose COCO detectors, not plate detectors. For actual ANPR use a plate-trained checkpoint.")

    @staticmethod
    def _crop(image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        return crop if crop.size else None

    def extract_text(self, image: np.ndarray, bbox: np.ndarray) -> str:
        crop = self._crop(image, bbox)
        if crop is None:
            return ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pieces = self.reader.readtext(gray, detail=0, paragraph=False)
        return " ".join(str(p).strip() for p in pieces if str(p).strip()).strip()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Received an empty or invalid image frame.")
        annotated = frame.copy()
        result = self.model.predict(source=frame, conf=self.confidence, device=self.device, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return annotated, detections
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            score = float(box.conf[0].item()) if box.conf is not None else 0.0
            name = self.names.get(class_id, str(class_id)) if isinstance(self.names, dict) else str(class_id)
            text = self.extract_text(frame, xyxy)
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
            label = f"{name} {score:.2f}" + (f" | {text}" if text else "")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(annotated, label[:100], (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
            detections.append({"class_name": name, "confidence": score, "bbox_xyxy": [x1, y1, x2, y2], "text": text})
        return annotated, detections

    def infer_image(self, source: str, output_path: str) -> None:
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}. Check the path and file format.")
        annotated, detections = self.process_frame(image)
        out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), annotated):
            raise OSError(f"Could not save output image: {out}")
        print(f"Saved image: {out.resolve()}\nDetections: {len(detections)}")
        for i, d in enumerate(detections, 1):
            print(f"{i}. class={d['class_name']}, confidence={d['confidence']:.3f}, OCR={d['text']!r}")

    def infer_video(self, source: PathLike = 0, output_path: Optional[str] = None, display: bool = True) -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video source: {source}. For webcam use --source 0; check file paths otherwise.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0: fps = 25.0
        writer = None
        if output_path:
            out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise OSError(f"Could not create output video: {out}. Try an .mp4 path.")
        print("Starting video inference. Press Q in the preview window to stop.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                annotated, _ = self.process_frame(frame)
                if writer is not None: writer.write(annotated)
                if display:
                    cv2.imshow("ANPR - press Q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
        finally:
            cap.release()
            if writer is not None: writer.release()
            if display: cv2.destroyAllWindows()
        if output_path: print(f"Saved video: {Path(output_path).resolve()}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="License-plate detection (YOLO) and OCR (EasyOCR).")
    p.add_argument("--model", default="models/number-plate.pt", help="Path to a plate-trained YOLO checkpoint.")
    p.add_argument("--source", default="input/car.mp4", help="Image/video path, or 0 for webcam.")
    p.add_argument("--output", default="output/anpr_output.mp4", help="Output image/video path.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "0"], help="auto, cpu, or NVIDIA CUDA device 0.")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold between 0 and 1.")
    p.add_argument("--no-display", action="store_true", help="Disable video preview window.")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    if not 0 <= args.conf <= 1:
        print("ERROR: --conf must be between 0 and 1.", file=sys.stderr); return 2
    webcam = args.source == "0"
    source = Path(args.source)
    if not webcam and not source.exists():
        print(f"ERROR: Input file not found: {args.source}. Put it in input/ or pass a valid path.", file=sys.stderr); return 2
    try:
        app = ANPR(args.model, device=args.device, confidence=args.conf)
        if webcam:
            app.infer_video(0, args.output, display=not args.no_display)
        elif source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            app.infer_image(str(source), args.output)
        else:
            app.infer_video(str(source), args.output, display=not args.no_display)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
