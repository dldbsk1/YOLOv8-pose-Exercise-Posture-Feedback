# export_to_onnx.py

from pathlib import Path
import shutil
import torch
import torch.nn as nn
from ultralytics import YOLO

from config import BASE_DIR, RUNS_DIR

# =========================
# 경로 설정
# =========================

YOLO_PT_PATH = RUNS_DIR / "pose" / "runs" / "pose" / "yolov8s_ph2_cf" / "weights" / "best.pt"

# 최종 배포용 ONNX 파일명
YOLO_ONNX_FINAL = BASE_DIR / "yolos_ph2_best.onnx"

LSTM_PT_PATH = BASE_DIR / "exercise_lstm_velo.pth"
LSTM_ONNX_FINAL = BASE_DIR / "exercise_lstm_velo.onnx"


# =========================
# 1) YOLO Pose ONNX 변환
# =========================

if not YOLO_PT_PATH.exists():
    raise FileNotFoundError(f"YOLO pt 파일이 없습니다: {YOLO_PT_PATH}")

print("[YOLO] PT:", YOLO_PT_PATH)

yolo_model = YOLO(str(YOLO_PT_PATH), task="pose")

exported_yolo_path = yolo_model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=True
)

exported_yolo_path = Path(exported_yolo_path)

shutil.copy2(exported_yolo_path, YOLO_ONNX_FINAL)

print("[YOLO] ONNX 변환 완료:", YOLO_ONNX_FINAL)


# =========================
# 2) LSTM ONNX 변환
# =========================

class ExerciseLSTM(nn.Module):
    def __init__(self, input_size=71, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


if not LSTM_PT_PATH.exists():
    raise FileNotFoundError(f"LSTM pth 파일이 없습니다: {LSTM_PT_PATH}")

print("[LSTM] PTH:", LSTM_PT_PATH)

device = torch.device("cpu")

lstm_model = ExerciseLSTM(
    input_size=71,
    hidden_size=64,
    num_layers=2,
    num_classes=4
).to(device)

state_dict = torch.load(
    LSTM_PT_PATH,
    map_location="cpu",
    weights_only=False
)

lstm_model.load_state_dict(state_dict)
lstm_model.eval()

dummy_input = torch.randn(1, 35, 71, dtype=torch.float32).to(device)

torch.onnx.export(
    lstm_model,
    dummy_input,
    str(LSTM_ONNX_FINAL),
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("[LSTM] ONNX 변환 완료:", LSTM_ONNX_FINAL)


# =========================
# 3) ONNX 파일 검증
# =========================

try:
    import onnx

    for path in [YOLO_ONNX_FINAL, LSTM_ONNX_FINAL]:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        print("[CHECK] ONNX OK:", path)

except Exception as e:
    print("[CHECK] ONNX 검증 실패:", e)
    raise

print("\n전체 ONNX 변환 완료")