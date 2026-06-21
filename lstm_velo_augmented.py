# %%
# -*- coding: utf-8 -*-

import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===================== 설정 =====================
TRAIN_CSV = "lstm_velo_data.csv"
MODEL_OUT = "exercise_lstm_velo.pth"
CONFIG_OUT = "lstm_velo_config.json"

SEQUENCE_LENGTH = 35
INPUT_SIZE = 71          # 좌표34 + 속도34 + 정적특징3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
STEP_SIZE = 1
LR = 0.0003
DROPOUT = 0.6
PATIENCE = 5

USE_WEIGHT = True
CLASS_NAMES = ["레그레이즈", "런지", "플랭크", "푸쉬업"]

VAL_VIDEOS_FIXED = [
    "leg_leg_01",
    "leg_leg_08",
    "lunge_lunge_05",
    "lunge_lunge_wait_01",
    "plank_plank_normal_01",
    "plank_plank_02",
    "pushup_pushup_dark_01",
    "pushup_pushup_03"
]

JOINT_WEIGHTS = np.ones(17, dtype=np.float32)
JOINT_WEIGHTS[[11, 12, 13, 14, 15, 16]] = 2.0
JOINT_WEIGHTS[[0, 1, 2, 3, 4]] = 0.5

SMOOTH_WIN = 3
VEL_GAP = 5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def coord_weight_vec():
    return np.repeat(JOINT_WEIGHTS, 2).astype(np.float32)


def smooth_coords(win34, k=SMOOTH_WIN):
    T = len(win34)
    if T < k or k <= 1:
        return win34.copy()

    out = win34.copy()
    half = k // 2

    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        out[t] = win34[s:e].mean(axis=0)

    return out


def make_features(win34, gap=VEL_GAP):
    """
    (T,34) → (T,71)
    좌표34 + 속도34 + 정적특징3
    정적특징:
      1) joint_var
      2) lower_var
      3) upper_speed_mean
    """
    win34 = np.asarray(win34, dtype=np.float32)

    if win34.shape[1] != 34:
        raise ValueError(f"win34는 (T,34)여야 합니다. 현재 shape: {win34.shape}")

    coords = smooth_coords(win34)
    T = len(coords)

    # 전체 관절 속도
    vel = np.zeros_like(coords)
    for t in range(T):
        prev = max(0, t - gap)
        vel[t] = coords[t] - coords[prev]

    # 전체 관절 분산
    joint_var = coords.var(axis=0).mean()

    # 하체 관절 분산
    lower_idx = []
    for j in [11, 12, 13, 14, 15, 16]:
        lower_idx += [j * 2, j * 2 + 1]

    lower_var = coords[:, lower_idx].var(axis=0).mean()

    # 상체 속도 평균: 어깨, 팔꿈치, 손목
    upper_idx = []
    for j in [5, 6, 7, 8, 9, 10]:
        upper_idx += [j * 2, j * 2 + 1]

    upper_vel = vel[:, upper_idx]                  # (T,12)
    upper_vel_xy = upper_vel.reshape(T, -1, 2)     # (T,6,2)
    upper_speed_mean = np.linalg.norm(
        upper_vel_xy,
        axis=2
    ).mean()

    static = np.tile(
        [joint_var, lower_var, upper_speed_mean],
        (T, 1)
    ).astype(np.float32)

    out = np.concatenate([win34, vel, static], axis=1)

    if out.shape[1] != INPUT_SIZE:
        raise ValueError(f"입력 차원 오류: {out.shape}, INPUT_SIZE={INPUT_SIZE}")

    return out


def feature_weight_vec():
    cw = coord_weight_vec()

    return np.concatenate([
        cw,
        cw,
        np.ones(3, dtype=np.float32)
    ])


def video_id(fname):
    base = os.path.splitext(str(fname))[0]
    parts = base.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    return base


def frame_no(fname):
    base = os.path.splitext(str(fname))[0]
    parts = base.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    return 0


def make_sequences_from_df(df, use_weight=USE_WEIGHT):
    feat = []
    for i in range(1, 18):
        feat += [f"x{i}", f"y{i}"]

    full_w = feature_weight_vec()

    X_seq = []
    y_seq = []

    for vid, g in df.groupby("_vid"):
        g = g.sort_values("_fno")

        X_raw = g[feat].values.astype(np.float32)
        y_raw = g["label"].values.astype(int)

        if len(X_raw) < SEQUENCE_LENGTH:
            continue

        for i in range(0, len(X_raw) - SEQUENCE_LENGTH + 1, STEP_SIZE):
            win = make_features(X_raw[i:i + SEQUENCE_LENGTH])

            if use_weight:
                win = win * full_w

            X_seq.append(win)

            label = np.bincount(
                y_raw[i:i + SEQUENCE_LENGTH],
                minlength=NUM_CLASSES
            ).argmax()

            y_seq.append(label)

    if len(X_seq) == 0:
        raise ValueError("생성된 시퀀스가 0개입니다.")

    X = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y = torch.tensor(np.array(y_seq), dtype=torch.long)

    return X, y


def prepare_by_video_split(csv_path, use_weight=USE_WEIGHT):
    df = pd.read_csv(csv_path)

    df["_vid"] = df["filename"].apply(video_id)
    df["_fno"] = df["filename"].apply(frame_no)

    all_vids = sorted(df["_vid"].unique().tolist())

    val_vids = VAL_VIDEOS_FIXED.copy()

    missing_val = [v for v in val_vids if v not in all_vids]
    if missing_val:
        print("[경고] CSV에 없는 VAL 영상:", missing_val)

    train_vids = [v for v in all_vids if v not in val_vids]

    overlap = set(train_vids) & set(val_vids)
    print("Train/Val 겹치는 영상:", overlap)

    print("\n영상 단위 분할")
    print("Train 영상 수:", len(train_vids))
    print("Val 영상 수:", len(val_vids))
    print("Val 영상 목록:", val_vids)

    train_df = df[df["_vid"].isin(train_vids)].copy()
    val_df = df[df["_vid"].isin(val_vids)].copy()

    X_train, y_train = make_sequences_from_df(train_df, use_weight=use_weight)
    X_val, y_val = make_sequences_from_df(val_df, use_weight=use_weight)

    return X_train, y_train, X_val, y_val, train_vids, val_vids


class ExerciseLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            INPUT_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )

        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    set_seed(SEED)

    print(f"학습 장치: {device}")

    X_train, y_train, X_val, y_val, train_vids, val_vids = prepare_by_video_split(
        TRAIN_CSV,
        use_weight=USE_WEIGHT
    )

    print(f"\nTrain 시퀀스: {X_train.shape}")
    print(f"Val 시퀀스: {X_val.shape}")

    counts = []
    print("\nTrain 클래스별 시퀀스 수")

    for c in range(NUM_CLASSES):
        n = (y_train == c).sum().item()
        counts.append(n)
        print(f"  {CLASS_NAMES[c]}: {n}개")

    counts_t = torch.tensor(counts, dtype=torch.float32)
    class_w = counts_t.sum() / (NUM_CLASSES * counts_t)
    class_w = class_w.to(device)

    print(f"클래스 가중치: {[round(w, 2) for w in class_w.tolist()]}")

    tr_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    va_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ExerciseLSTM().to(device)

    crit = nn.CrossEntropyLoss(weight=class_w)

    opt = optim.Adam(model.parameters(), lr=LR)

    print("\n학습 시작 (video_id 기준 val 분리 + 상체 속도 특징 추가)")

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()

        tr_loss = 0.0
        correct = 0
        total = 0

        for bx, by in tr_loader:
            bx = bx.to(device)
            by = by.to(device)

            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)

            loss.backward()
            opt.step()

            tr_loss += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)

        model.eval()

        v_loss = 0.0
        v_correct = 0
        v_total = 0

        with torch.no_grad():
            for bx, by in va_loader:
                bx = bx.to(device)
                by = by.to(device)

                out = model(bx)
                loss = crit(out, by)

                v_loss += loss.item() * bx.size(0)
                v_correct += (out.argmax(1) == by).sum().item()
                v_total += by.size(0)

        tr_loss = tr_loss / total
        tr_acc = correct / total * 100

        v_loss = v_loss / v_total
        v_acc = v_correct / v_total * 100

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"train_loss:{tr_loss:.4f} acc:{tr_acc:.1f}% | "
            f"val_loss:{v_loss:.4f} val_acc:{v_acc:.1f}%"
        )

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_cnt = 0
        else:
            patience_cnt += 1

            if patience_cnt >= PATIENCE:
                print(f"조기종료 (epoch {epoch + 1}, val_loss 개선 없음)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_OUT)

    config = {
        "sequence_length": SEQUENCE_LENGTH,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES,
        "dropout": DROPOUT,
        "use_weight": USE_WEIGHT,
        "class_names": CLASS_NAMES,
        "joint_weights": JOINT_WEIGHTS.tolist(),
        "smooth_win": SMOOTH_WIN,
        "vel_gap": VEL_GAP,
        "feature_type": "coords_velocity_variance_upper_speed",
        "train_videos": train_vids,
        "val_videos": val_vids
    }

    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {MODEL_OUT}, {CONFIG_OUT}")


if __name__ == "__main__":
    main()