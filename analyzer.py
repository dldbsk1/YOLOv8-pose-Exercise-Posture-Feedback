# -*- coding: utf-8 -*-
"""
영상/웹캠 분석 모듈
- YOLOv8 Pose ONNX로 관절 추출
- LSTM ONNX로 운동 분류
- pose_rules로 자세 평가
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from config import YOLO_POSE_ONNX, LSTM_ONNX
from pose_rules import evaluate_pose


# ===================== 설정 =====================
INPUT_SIZE = 71
SEQUENCE_LENGTH = 35
NUM_CLASSES = 4

CLASS_NAMES = ["레그레이즈", "런지", "플랭크", "푸쉬업"]

IMGSZ = 640
DET_CONF = 0.25
KP_CONF = 0.5

FRAME_STRIDE = 2
BATCH_SIZE = 1
SEQ_STEP = 10

OE_MIN_CUTOFF = 1.0
OE_BETA = 0.007
OE_DCUTOFF = 1.0

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ===================== One-Euro Filter =====================
class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, dcutoff=1.0, freq=30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq = freq
        self.x_filt = None
        self.dx_filt = None
        self.prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, kpts):
        kpts = kpts.astype(np.float32)

        if self.x_filt is None:
            n = kpts.size
            self.x_filt = [_LowPass() for _ in range(n)]
            self.dx_filt = [_LowPass() for _ in range(n)]
            self.prev = kpts.flatten()

        flat = kpts.flatten()
        out = np.empty_like(flat)

        for i, x in enumerate(flat):
            if x == 0.0:
                out[i] = 0.0
                self.prev[i] = 0.0
                continue

            dx = (x - self.prev[i]) * self.freq
            edx = self.dx_filt[i](dx, self._alpha(self.dcutoff, self.freq))
            cutoff = self.min_cutoff + self.beta * abs(edx)
            out[i] = self.x_filt[i](x, self._alpha(cutoff, self.freq))
            self.prev[i] = x

        return out.reshape(kpts.shape)


# ===================== 유틸 =====================
def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _reencode_h264(src, dst):
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                dst
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def draw_skeleton_on(frame, kpts_px):
    for i, j in SKELETON:
        pi, pj = kpts_px[i], kpts_px[j]
        if np.linalg.norm(pi) > 1 and np.linalg.norm(pj) > 1:
            cv2.line(
                frame,
                (int(pi[0]), int(pi[1])),
                (int(pj[0]), int(pj[1])),
                (200, 200, 200),
                2
            )

    for x, y in kpts_px:
        if x > 1 and y > 1:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 200, 255), -1)

    return frame


def _pick_main(result):
    if result.keypoints is None or result.keypoints.xy is None:
        return None, None

    xy = result.keypoints.xy.cpu().numpy()

    if xy.shape[0] == 0:
        return None, None

    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()
    else:
        conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    if xy.shape[0] == 1:
        idx = 0
    elif result.boxes is not None:
        areas = [b[2] * b[3] for b in result.boxes.xywh.cpu().numpy()]
        idx = int(np.argmax(areas))
    else:
        idx = 0

    return xy[idx], conf[idx]


def _mask_low_conf(xy, conf, thr):
    out = xy.copy()
    out[conf < thr] = 0.0
    return out


def _normalize(xy, w, h):
    n = xy.copy().astype(np.float32)
    n[:, 0] /= max(w, 1)
    n[:, 1] /= max(h, 1)
    return n


def _dist(a, b):
    if np.linalg.norm(a) <= 0 or np.linalg.norm(b) <= 0:
        return 0.0
    return float(np.linalg.norm(a - b))


def _static_features(norm):
    shoulder_width = _dist(norm[5], norm[6])
    hip_width = _dist(norm[11], norm[12])

    valid = norm[np.linalg.norm(norm, axis=1) > 0]
    if len(valid) > 0:
        body_height = float(valid[:, 1].max() - valid[:, 1].min())
    else:
        body_height = 0.0

    return np.array([shoulder_width, hip_width, body_height], dtype=np.float32)


def _make_lstm_feature_sequence(chunk):
    """
    chunk: list of (17, 2) normalized keypoints
    output: (35, 71)
    71 = 좌표 34 + 속도 34 + 정적 특징 3
    """
    coords = np.array([c.flatten() for c in chunk], dtype=np.float32)

    velocity = np.zeros_like(coords, dtype=np.float32)
    velocity[1:] = coords[1:] - coords[:-1]

    statics = np.array([_static_features(c) for c in chunk], dtype=np.float32)

    features = np.concatenate([coords, velocity, statics], axis=1)

    return features.astype(np.float32)


def _get_ort_providers():
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


# ===================== 모델 로드 =====================
_models = {}


def load_models():
    """
    YOLO ONNX + LSTM ONNX 로드
    """
    if _models:
        return _models["pose"], _models["lstm"], _models["lstm_input"], _models["providers"]

    yolo_path = str(YOLO_POSE_ONNX)
    lstm_path = str(LSTM_ONNX)

    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO ONNX 파일이 없습니다: {yolo_path}")

    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM ONNX 파일이 없습니다: {lstm_path}")

    pose_model = YOLO(yolo_path, task="pose")
    
    providers = _get_ort_providers()
    lstm_session = ort.InferenceSession(lstm_path, providers=providers)
    lstm_input_name = lstm_session.get_inputs()[0].name

    _models["pose"] = pose_model
    _models["lstm"] = lstm_session
    _models["lstm_input"] = lstm_input_name
    _models["providers"] = providers

    print("[MODEL] YOLO ONNX:", yolo_path)
    print("[MODEL] LSTM ONNX:", lstm_path)
    print("[ONNXRuntime Providers]:", providers)

    return pose_model, lstm_session, lstm_input_name, providers


# ===================== 메인 분석 =====================
def analyze_video(source, progress_cb=None, skeleton_out=None, debug=False, det_conf=None):
    conf_thr = DET_CONF if det_conf is None else float(det_conf)

    pose_model, lstm_session, lstm_input_name, providers = load_models()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {source}")

    sampled = []
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if fidx % FRAME_STRIDE == 0:
            h, w = frame.shape[:2]
            sampled.append((frame, w, h))

        fidx += 1

    cap.release()

    px_frames = []
    norm_frames = []

    valid_frames = 0
    conf_accum = np.zeros(17, dtype=np.float64)
    conf_count = 0

    n = len(sampled)

    for b in range(0, n, BATCH_SIZE):
        batch = sampled[b:b + BATCH_SIZE]
        frames = [item[0] for item in batch]

        results = pose_model.predict(
            frames,
            conf=conf_thr,
            imgsz=IMGSZ,
            verbose=False
        )

        for (frame, w, h), res in zip(batch, results):
            xy, conf = _pick_main(res)

            if xy is None:
                px_frames.append(None)
                norm_frames.append(None)
                continue

            conf_accum += conf
            conf_count += 1

            masked = _mask_low_conf(xy, conf, KP_CONF)
            px_frames.append(masked.astype(np.float32))
            norm_frames.append(_normalize(masked, w, h))

            valid_frames += 1

        if progress_cb and n:
            progress_cb(min((b + len(batch)) / n, 1.0) * 0.85)

    kp_conf_mean = (conf_accum / conf_count).tolist() if conf_count else [0] * 17

    if debug and conf_count:
        kp_names = [
            "코", "왼눈", "오눈", "왼귀", "오귀",
            "왼어깨", "오어깨", "왼팔꿈치", "오팔꿈치",
            "왼손목", "오손목", "왼골반", "오골반",
            "왼무릎", "오무릎", "왼발목", "오발목"
        ]

        print("\n===== 관절별 평균 신뢰도 =====")
        for i, name in enumerate(kp_names):
            print(f"{name:8s}: {kp_conf_mean[i]:.3f}")
        print("=" * 40)

    # 스무딩
    oe = OneEuroFilter(
        OE_MIN_CUTOFF,
        OE_BETA,
        OE_DCUTOFF,
        freq=30.0 / FRAME_STRIDE
    )

    for i in range(len(px_frames)):
        if px_frames[i] is None:
            continue

        smoothed = oe(px_frames[i])
        px_frames[i] = smoothed

        _, w, h = sampled[i]
        norm_frames[i] = _normalize(smoothed, w, h)

    # 스켈레톤 영상 저장
    skeleton_video = None

    if skeleton_out and sampled:
        _, w0, h0 = sampled[0]
        fps_out = max(30.0 / FRAME_STRIDE, 1.0)

        writer = None
        used_codec = "mp4v"

        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(skeleton_out, fourcc, fps_out, (w0, h0))

            if writer.isOpened():
                used_codec = codec
                break

            writer.release()

        for i, (frame, w, h) in enumerate(sampled):
            canvas = frame.copy()

            if px_frames[i] is not None:
                draw_skeleton_on(canvas, px_frames[i])

            writer.write(canvas)

        writer.release()

        skeleton_video = skeleton_out

        if used_codec == "mp4v":
            h264_path = os.path.splitext(skeleton_out)[0] + "_h264.mp4"

            if _reencode_h264(skeleton_out, h264_path):
                skeleton_video = h264_path

    # ===================== LSTM ONNX 분류 =====================
    valid_norm = [f for f in norm_frames if f is not None]

    seqs = []

    for i in range(0, len(valid_norm) - SEQUENCE_LENGTH + 1, SEQ_STEP):
        chunk = valid_norm[i:i + SEQUENCE_LENGTH]
        feat = _make_lstm_feature_sequence(chunk)

        if feat.shape == (SEQUENCE_LENGTH, INPUT_SIZE):
            seqs.append(feat)

    if seqs:
        batch = np.stack(seqs, axis=0).astype(np.float32)

        logits = lstm_session.run(
            None,
            {lstm_input_name: batch}
        )[0]

        probs = _softmax(logits)

        mean_prob = np.mean(probs, axis=0)
        cls = int(np.argmax(mean_prob))

        exercise = CLASS_NAMES[cls]
        confidence = float(mean_prob[cls]) * 100
        n_sequences = len(seqs)

    else:
        exercise = CLASS_NAMES[0]
        confidence = 0.0
        n_sequences = 0

    # ===================== 자세 평가 =====================
    item_ok = {}
    item_bad_desc = {}
    item_weight = {}

    phase_count = {"down": 0, "up": 0, "none": 0}
    phase_seq = []

    for norm in norm_frames:
        if norm is None:
            phase_seq.append("none")
            phase_count["none"] += 1
            continue

        results, phase = evaluate_pose(exercise, norm)

        phase_count[phase] = phase_count.get(phase, 0) + 1
        phase_seq.append(phase)

        for r in results:
            item_ok.setdefault(r["label"], []).append(1 if r["ok"] else 0)
            item_weight[r["label"]] = r.get("weight", 1)

            if not r["ok"]:
                item_bad_desc.setdefault(r["label"], []).append(r["desc"])

    item_scores = []

    for label, oks in item_ok.items():
        ratio = sum(oks) / len(oks)

        item_scores.append({
            "label": label,
            "score": int(ratio * 100),
            "n": len(oks),
            "weight": item_weight.get(label, 1),
        })

    item_scores.sort(key=lambda x: x["score"])

    if item_scores:
        total_weight = sum(s["weight"] for s in item_scores)
        avg_score = int(
            sum(s["score"] * s["weight"] for s in item_scores) / total_weight
        ) if total_weight else 0
    else:
        avg_score = 0

    def _most_common(lst):
        if not lst:
            return None

        counts = {}

        for item in lst:
            counts[item] = counts.get(item, 0) + 1

        return max(counts, key=counts.get)

    feedbacks = []

    for s in item_scores:
        label = s["label"]
        score = s["score"]
        common_desc = _most_common(item_bad_desc.get(label, []))

        if score >= 80:
            title = "✅ " + label
            text = "대부분 잘 수행되고 있어요. 지금처럼 유지하세요!"
        elif score >= 50:
            title = "⚠️ " + label
            text = "조금만 더 다듬으면 될 것 같아요."

            if common_desc:
                text += f" {common_desc.split('(')[0].strip()}"
        else:
            title = "❌ " + label
            text = "이 부분은 교정이 필요해요."

            if common_desc:
                text += f" {common_desc.split('(')[0].strip()}"

        feedbacks.append((title, text))

    # ===================== 횟수 / 유지 시간 =====================
    eff_fps = max(30.0 / FRAME_STRIDE, 1.0)

    rep_count = None
    hold_seconds = None

    if exercise == "플랭크":
        hold_frames = phase_count.get("down", 0)
        hold_seconds = round(hold_frames / eff_fps, 1)

    else:
        min_hold = max(int(eff_fps * 0.2), 2)
        rep_count = 0

        stable_phase = "up"
        run_phase = None
        run_len = 0

        for ph in phase_seq:
            if ph == "none":
                run_phase = None
                run_len = 0
                continue

            if ph == run_phase:
                run_len += 1
            else:
                run_phase = ph
                run_len = 1

            if run_len >= min_hold and ph != stable_phase:
                if stable_phase == "up" and ph == "down":
                    rep_count += 1

                stable_phase = ph

    if progress_cb:
        progress_cb(1.0)

    return {
        "exercise": exercise,
        "confidence": round(confidence, 1),
        "n_sequences": n_sequences,
        "valid_frames": valid_frames,
        "avg_score": avg_score,
        "item_scores": item_scores,
        "feedbacks": feedbacks,
        "skeleton_video": skeleton_video,
        "kp_conf_mean": kp_conf_mean,
        "rep_count": rep_count,
        "hold_seconds": hold_seconds,
    }