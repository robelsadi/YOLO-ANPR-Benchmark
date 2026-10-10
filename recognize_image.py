
import cv2
import easyocr
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# ANPR: Automatic Number Plate Recognition
# YOLO26 + EasyOCR
# ==========================================

# 1. Define file paths
image_path = Path("input/car.jpg")
model_path = Path("models/yolo26m_face_lp.pt")
output_path = Path("output/car_recognized.jpg")

# 2. Check that the required files exist
if not image_path.exists():
    raise FileNotFoundError(f"Input image not found: {image_path}")

if not model_path.exists():
    raise FileNotFoundError(f"YOLO model not found: {model_path}")

# 3. Load the input image
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Could not read image: {image_path}")

annotated = image.copy()

# 4. Load the YOLO model
print("Loading YOLO model...")
model = YOLO(str(model_path))

# 5. Detect objects in the image
print("Detecting license plates...")
result = model.predict(
    source=image,
    conf=0.25,
    device="cpu",
    verbose=False
)[0]

# 6. Initialize EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

plate_count = 0

# 7. Process detected license plates
for box in result.boxes:
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]

    # Ignore faces and process only license plates
    if class_name != "license-plate":
        continue

    # Get the bounding-box coordinates
    x1, y1, x2, y2 = map(
        int, box.xyxy[0].cpu().numpy()
    )

    # Keep coordinates within the image boundaries
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        continue

    # Crop the detected number plate
    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        continue

    # Prepare the crop for OCR
    gray_plate = cv2.cvtColor(
        plate_crop, cv2.COLOR_BGR2GRAY
    )

    # Recognize the characters
    recognized_words = reader.readtext(
        gray_plate,
        detail=0,
        paragraph=False
    )

    plate_text = " ".join(recognized_words).strip()
    confidence = float(box.conf[0].item())
    plate_count += 1

    print(f"\nPlate {plate_count}: {plate_text or 'No text recognized'}")
    print(f"Detection confidence: {confidence:.3f}")

    # Draw a green rectangle around the number plate
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Write the recognized text above the rectangle
    label = plate_text if plate_text else "No text recognized"

    # Add a solid background to improve text readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    text_y = max(text_height + 10, y1 - 10)

    cv2.rectangle(
        annotated,
        (x1, text_y - text_height - 8),
        (min(width - 1, x1 + text_width + 8), text_y + baseline),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        annotated,
        label,
        (x1 + 4, text_y - 3),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

# 8. Save the annotated image
output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), annotated):
    raise OSError(f"Could not save output image: {output_path}")

print("\n================================")
print(f"Plates detected: {plate_count}")
print(f"Saved image: {output_path.resolve()}")
print("================================")

# 9. Display the image in a pop-up window
if plate_count == 0:
    print("No license plates were detected in this image.")

window_name = "ANPR - License Plate Recognition"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, annotated)

print("\nThe result image is displayed.")
print("Click the image window and press any key to close it.")

# Keep the pop-up visible until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
