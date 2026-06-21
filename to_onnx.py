# export_to_onnx.py

import torch
from ultralytics import YOLO

# =========================
# 1) YOLO Pose ONNX 변환
# =========================

YOLO_PT_PATH = r"C:\Users\DS\Desktop\cv\runs\pose\runs\pose\yolov8s_ph2_cf\weights\best.pt"

yolo_model = YOLO(YOLO_PT_PATH)

yolo_model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False
)

print("YOLO Pose ONNX 변환 완료")


# =========================
# 2) LSTM ONNX 변환
# =========================
# 주의: 아래 LSTM 클래스 구조는 네가 학습할 때 쓴 모델 구조와 동일해야 함

import torch.nn as nn


class ExerciseLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, num_classes=4):
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


LSTM_PT_PATH = r"C:\Users\DS\Desktop\cv\exercise_lstm_velo.pth"
LSTM_ONNX_PATH = r"C:\Users\DS\Desktop\cv\exercise_lstm_velo.onnx"

device = torch.device("cpu")

lstm_model = ExerciseLSTM(
    input_size=71,
    hidden_size=64,
    num_layers=2,
    num_classes=4
).to(device)

state_dict = torch.load(LSTM_PT_PATH, map_location="cpu")
lstm_model.load_state_dict(state_dict)
lstm_model.eval()

dummy_input = torch.randn(1, 35, 71).to(device)
# 1개 영상, 30프레임 시퀀스, 34차원 좌표

torch.onnx.export(
    lstm_model,
    dummy_input,
    LSTM_ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("LSTM ONNX 변환 완료")