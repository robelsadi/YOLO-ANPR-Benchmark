import importlib, sys
print("Python:", sys.version)
failed = []
for name in ["cv2", "torch", "easyocr", "ultralytics", "numpy"]:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'version not exposed')}")
    except Exception as exc:
        failed.append(name); print(f"[FAIL] {name}: {exc}")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
except Exception: pass
if failed:
    print("\nOne or more imports failed. Follow README.md."); raise SystemExit(1)
print("\nEnvironment checks passed.")
