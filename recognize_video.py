import cv2
import easyocr
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "input" / "sample_30fps_1440.mp4"
MODEL_PATH = BASE_DIR / "models" / "yolo26m_face_lp.pt"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "video_960_fixed.mp4"

# Detection settings
IMAGE_SIZE = 960       # Later compare 640 vs 960
CONFIDENCE = 0.25
PLATE_CLASS_ID = 1     # Model class: license-plate

# ==========================================================
# 2. CHECK GPU
# ==========================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
    print(f"GPU enabled: {DEVICE_NAME}")
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
    print("CUDA is unavailable. Using CPU.")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ==========================================================
# 3. CHECK REQUIRED FILES
# ==========================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# ==========================================================
# 4. LOAD YOLO AND OCR
# ==========================================================

print("\nLoading YOLO model...")
model = YOLO(str(MODEL_PATH))

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

# OCR remains on CPU initially to conserve GPU memory.
# YOLO detection will use the NVIDIA GPU when available.

# ==========================================================
# 5. OPEN INPUT VIDEO
# ==========================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_fps = fps if fps > 0 else 25.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    output_fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

print("\n========== ANPR EXPERIMENT ==========")
print(f"Input video: {VIDEO_PATH.name}")
print(f"Resolution: {width} x {height}")
print(f"Source FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"YOLO image size: {IMAGE_SIZE}")
print(f"Detection confidence threshold: {CONFIDENCE}")
print(f"Output video: {OUTPUT_PATH.name}")
print("Press Q in the video window to stop early.\n")

# ==========================================================
# 6. PROCESS VIDEO
# ==========================================================

frame_number = 0
frames_processed = 0
plates_detected = 0
plates_with_text = 0

start_time = time.perf_counter()

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # YOLO license-plate detection on GPU
        results = model.predict(
            source=frame,
            classes=[PLATE_CLASS_ID],
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Keep coordinates inside the frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                plates_detected += 1
                confidence = float(box.conf[0])

                # OCR on CPU
                ocr_results = reader.readtext(
                    plate_crop,
                    detail=0
                )

                plate_text = " ".join(
                    text.strip() for text in ocr_results
                    if text.strip()
                )

                if plate_text:
                    plates_with_text += 1
                    print(
                        f"Frame {frame_number}: "
                        f"{plate_text} | "
                        f"Confidence: {confidence:.2f}"
                    )

                label = plate_text if plate_text else "Plate detected"
                label = f"{label} ({confidence:.2f})"

                # Draw green plate bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw label background
                label_y = max(y1 - 32, 0)
                label_width = min(width - x1, 450)

                cv2.rectangle(
                    frame,
                    (x1, label_y),
                    (x1 + label_width, y1),
                    (0, 150, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    label[:55],
                    (x1 + 5, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

        # Save processed frame
        writer.write(frame)
        frames_processed += 1

        # Show live preview
        cv2.imshow("ANPR - GPU Video Recognition", frame)

        # Progress update every 30 frames
        if frame_number % 30 == 0:
            elapsed = time.perf_counter() - start_time
            average_fps = frames_processed / elapsed if elapsed > 0 else 0

            print(
                f"Progress: {frame_number}/{total_frames} frames | "
                f"Average processing FPS: {average_fps:.2f}"
            )

        # Press Q to stop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped early by user.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# ==========================================================
# 7. EXPERIMENT SUMMARY
# ==========================================================

total_time = time.perf_counter() - start_time
average_fps = (
    frames_processed / total_time
    if total_time > 0 else 0
)


# Video processing loop finishes above this point

writer.release()
cap.release()

print("\n========== EXPERIMENT SUMMARY ==========")
print(f"Device: {DEVICE_NAME}")

print(f"Frames processed: {frames_processed}")
print(f"Detection events: {plates_detected}")
print(f"Detection events with OCR text: {plates_with_text}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average processing FPS: {average_fps:.2f}")
print(f"Output video saved to: {OUTPUT_PATH}")
print("========================================")
