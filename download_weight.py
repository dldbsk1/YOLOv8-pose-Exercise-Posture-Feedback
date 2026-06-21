from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FILES = {
    "yolos_ph2_best.onnx": "https://drive.google.com/uc?export=download&id=1fLu4qYse7poO6cb6jGgMaYmvJowxdMRC",
    "exercise_lstm_velo.onnx": "https://drive.google.com/uc?export=download&id=1yUWUXh5acNg2qs7VQ-yjYn1_UVothj06",
}

for filename, url in FILES.items():
    save_path = MODELS_DIR / filename
    if not save_path.exists():
        print(f"[다운로드] {filename}")
        urllib.request.urlretrieve(url, save_path)
    else:
        print(f"[존재함] {filename}")