# ANPR with YOLO + EasyOCR (Windows / VS Code)

Detect plate regions with YOLO and recognize text from each crop with EasyOCR.

## Important model requirement
`yolo26n.pt` is an official general-purpose COCO detector, not a license-plate detector. Actual ANPR requires YOLO weights trained to detect license plates. Put the trained checkpoint at `models/number-plate.pt`. The code intentionally reports a clear error if that file is missing; do not rename unrelated weights to make the error disappear.

## 1. Open the project
Extract the ZIP and open the `anpr_vscode_project` folder in VS Code. Install the Microsoft Python extension. Use **Terminal → New Terminal**.

## 2. Create and activate an environment (PowerShell)
Python 3.10 or 3.11 is a practical choice. If Python 3.11 is installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked in this terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Then verify:

```powershell
python test_setup.py
```

If PyTorch has a platform-specific install issue, use the official PyTorch selector: https://pytorch.org/get-started/locally/ . CPU inference is supported; CUDA is optional. Do not install a separate CUDA Toolkit just to start this project.

## 4. Add the model and input

```text
anpr_vscode_project/
├── anpr.py
├── test_setup.py
├── requirements.txt
├── models/number-plate.pt  # your plate-trained checkpoint
├── input/car.mp4           # your video
└── output/                 # output files appear here
```

## 5. Run video, image, or webcam

Video:

```powershell
python anpr.py --model models/number-plate.pt --source input/car.mp4 --output output/anpr_output.mp4
```

Image:

```powershell
python anpr.py --model models/number-plate.pt --source input/car.jpg --output output/anpr_output.jpg
```

Webcam:

```powershell
python anpr.py --model models/number-plate.pt --source 0 --output output/webcam_output.mp4
```

Force CPU:

```powershell
python anpr.py --model models/number-plate.pt --source input/car.mp4 --output output/anpr_output.mp4 --device cpu
```

Process video without preview:

```powershell
python anpr.py --model models/number-plate.pt --source input/car.mp4 --output output/anpr_output.mp4 --no-display
```

## Common issues
- **Model file not found:** place plate-trained weights at `models/number-plate.pt` or change `--model` to the real path.
- **Input file not found:** check filename and extension, or pass a full path in quotes.
- **No plates detected:** confirm the checkpoint is a plate detector; try `--conf 0.15` cautiously.
- **EasyOCR first run:** internet may be needed to download English OCR weights.
- **CUDA error:** run with `--device cpu` first.

OCR is not guaranteed accurate with blur, glare, angled plates, small text, or unusual plate alphabets. This program performs inference only; it does not train a detector.
