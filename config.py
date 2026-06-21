# config.py
from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent

# -------------------------
# 폴더 경로
# -------------------------
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset-cf"
RUNS_DIR = BASE_DIR / "runs"

# -------------------------
# 데이터셋 경로
# -------------------------
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
VAL_IMAGES_DIR   = DATASET_DIR / "images" / "val"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"
VAL_LABELS_DIR   = DATASET_DIR / "labels" / "val"

# -------------------------
# YOLO 모델 경로
# -------------------------
YOLO_POSE_PT = MODELS_DIR / "yolos_ph2_best.pt"
YOLO_POSE_ONNX = MODELS_DIR / "yolos_ph2_best.onnx"

# -------------------------
# LSTM 모델 경로
# -------------------------
LSTM_PTH = MODELS_DIR / "exercise_lstm_velo.pth"
LSTM_ONNX = MODELS_DIR / "exercise_lstm_velo.onnx"
LSTM_CONFIG = MODELS_DIR / "lstm_velo_config.json"

# -------------------------
# 기타 파일
# -------------------------
FONT_PATH = ASSETS_DIR / "Malgun.ttf"

# -------------------------
# 클래스 / 모델 설정
# -------------------------
CLASS_NAMES = ["레그레이즈", "런지", "플랭크", "푸쉬업"]

NUM_CLASSES = 4
LSTM_INPUT_SIZE = 71
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
SEQUENCE_LENGTH = 35

YOLO_IMGSZ = 640
YOLO_CONF = 0.35