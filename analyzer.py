# -*- coding: utf-8 -*-
"""
영상/웹캠 분석 모듈
  - YOLOv8 pose 로 관절 추출
  - LSTM 으로 운동 분류
  - pose_rules 로 프레임별 자세 평가 → 전체 프레임 평균 점수 / 항목별 평균
Streamlit(app.py)에서 import 해서 사용.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from config import YOLO_POSE_ONNX

from pose_rules import evaluate_pose, EXERCISE_RULES


# ── One-Euro 필터 (관절 좌표 시간축 스무딩) ──
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
    """관절 (17,2) 배열을 프레임마다 스무딩. 각 좌표 독립 필터."""
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
            # 관절 수 x 2 만큼 필터 준비
            n = kpts.size
            self.x_filt = [_LowPass() for _ in range(n)]
            self.dx_filt = [_LowPass() for _ in range(n)]
            self.prev = kpts.flatten()

        flat = kpts.flatten()
        out = np.empty_like(flat)
        for i, x in enumerate(flat):
            # 0좌표(미검출 마스킹)는 그대로 통과 (필터 오염 방지)
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

# ── 설정 ──
INPUT_SIZE = 34
SEQUENCE_LENGTH = 30
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_CLASSES = 4
CLASS_NAMES = ["레그레이즈", "런지", "플랭크", "푸쉬업"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_WEIGHT_PATH = os.path.join(BASE_DIR, "exercise_lstm.pth")

# YOLO Pose 모델: 사전학습 가중치 사용
YOLO_POSE_WEIGHT = YOLO(str(YOLO_POSE_ONNX))
_local = os.path.join(BASE_DIR, YOLO_POSE_WEIGHT)
YOLO_POSE_WEIGHT_PATH = _local if os.path.exists(_local) else YOLO_POSE_WEIGHT

IMGSZ = 640
DET_CONF = 0.25
KP_CONF = 0.5

# 속도 옵션
FRAME_STRIDE = 2      # N프레임당 1장만 추론 (1=전체, 2=절반, 3=1/3)
                      #  주의: LSTM은 30프레임 시퀀스 기준 학습됨.
                      #  분류가 부정확해지면 1로 두세요(점수 집계엔 영향 적음).
BATCH_SIZE = 8        # 한 번에 추론할 프레임 수 (GPU면 16~32도 가능)

# LSTM 분류 옵션
SEQ_STEP = 10         # 시퀀스 슬라이딩 간격. 작을수록 시퀀스 많아져 평균 안정
                      #  (30=겹침없음, 10=20프레임 겹침, 1=학습과 동일하게 촘촘)

# One-Euro 필터 파라미터 (관절 좌표 스무딩)
OE_MIN_CUTOFF = 1.0   # 작을수록 더 부드럽지만 지연↑
OE_BETA = 0.007       # 빠른 움직임에 반응하는 정도
OE_DCUTOFF = 1.0

# 스켈레톤 연결 (COCO 17 keypoints)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _reencode_h264(src, dst):
    """ffmpeg로 H.264 재인코딩 (브라우저 재생용). 성공 시 True."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
            check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def draw_skeleton_on(frame, kpts_px):
    """프레임에 관절점+스켈레톤 그리기 (픽셀 좌표). 0좌표는 미검출로 스킵."""
    for i, j in SKELETON:
        pi, pj = kpts_px[i], kpts_px[j]
        if np.linalg.norm(pi) > 1 and np.linalg.norm(pj) > 1:
            cv2.line(frame, (int(pi[0]), int(pi[1])),
                     (int(pj[0]), int(pj[1])), (200, 200, 200), 2)
    for (x, y) in kpts_px:
        if x > 1 and y > 1:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 200, 255), -1)
    return frame


# ── LSTM 모델 ──
class ExerciseLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=32, num_layers=2, num_classes=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


_models = {}


def load_models():
    """YOLO + LSTM 로드 (1회만)"""
    if _models:
        return _models["pose"], _models["lstm"], _models["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = YOLO(YOLO_POSE_WEIGHT_PATH)
    lstm_model = ExerciseLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    if not os.path.exists(LSTM_WEIGHT_PATH):
        raise FileNotFoundError(f"{LSTM_WEIGHT_PATH} 가 없습니다.")
    state = torch.load(LSTM_WEIGHT_PATH, map_location=device)
    lstm_model.load_state_dict(state)
    lstm_model.eval()
    _models["pose"], _models["lstm"], _models["device"] = pose_model, lstm_model, device
    return pose_model, lstm_model, device


# ── 관절 추출 유틸 ──
def _pick_main(result):
    if result.keypoints is None or result.keypoints.xy is None:
        return None, None
    xy = result.keypoints.xy.cpu().numpy()
    if xy.shape[0] == 0:
        return None, None
    conf = (result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones((xy.shape[0], xy.shape[1]), np.float32))
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


# ── 메인 분석 ──
def analyze_video(source, progress_cb=None, skeleton_out=None, debug=False,
                  det_conf=None):
    """
    source: 영상 경로(str)
    progress_cb: 진행률 콜백 (0.0~1.0)
    skeleton_out: 스켈레톤 그린 영상 저장 경로(mp4). None이면 저장 안 함.
    debug: True면 관절별 신뢰도를 콘솔에 출력 (무릎/발목 진단용)
    det_conf: YOLO 검출 임계값. None이면 기본값 DET_CONF 사용.
              (Streamlit 사이드바 슬라이더 값이 여기로 전달됨)
    returns: dict {
        exercise, confidence, n_sequences, valid_frames,
        avg_score, item_scores, feedbacks, skeleton_video, kp_conf_mean
    }
    """
    # 슬라이더 값(det_conf)이 들어오면 그 값을, 없으면 모듈 기본값 사용
    conf_thr = DET_CONF if det_conf is None else float(det_conf)

    pose_model, lstm_model, device = load_models()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {source}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    seq_buffer = []          # LSTM 입력용 (정규화 34차원)
    seq_logits = []          # 시퀀스별 분류 결과
    valid_frames = 0
    # 항목별 통과 집계: {label: [ok(0/1), ...]}
    item_ok = {}
    # 항목별 '틀렸을 때'의 구체적 설명(desc) 모음: {label: [desc, ...]}
    item_bad_desc = {}
    # 관절별 신뢰도 누적 (마스킹 전 원본 conf)
    conf_accum = np.zeros(17, dtype=np.float64)
    conf_count = 0

    # ── 1) 프레임 샘플링: FRAME_STRIDE 간격으로만 읽어서 모음 ──
    sampled = []      # (frame, w, h)
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

    # ── 2) 배치 추론 ──
    px_frames = []      # 픽셀 좌표 (LSTM용)
    norm_frames = []    # 정규화 좌표 (규칙용)
    n = len(sampled)
    for b in range(0, n, BATCH_SIZE):
        batch = sampled[b:b + BATCH_SIZE]
        frames = [item[0] for item in batch]
        results = pose_model.predict(frames, conf=conf_thr,
                                     imgsz=IMGSZ, verbose=False)
        for (frame, w, h), res in zip(batch, results):
            xy, conf = _pick_main(res)
            if xy is None:
                px_frames.append(None)
                norm_frames.append(None)
            else:
                conf_accum += conf          # 마스킹 전 원본 신뢰도 누적
                conf_count += 1
                masked = _mask_low_conf(xy, conf, KP_CONF)
                px_frames.append(masked.astype(np.float32))
                norm_frames.append(_normalize(masked, w, h))
                valid_frames += 1
        if progress_cb and n:
            progress_cb(min((b + len(batch)) / n, 1.0) * 0.85)

    # ── 관절별 평균 신뢰도 (진단) ──
    KP_NAMES = ["코", "왼눈", "오눈", "왼귀", "오귀",
                "왼어깨", "오어깨", "왼팔꿈치", "오팔꿈치",
                "왼손목", "오손목", "왼골반", "오골반",
                "왼무릎", "오무릎", "왼발목", "오발목"]
    kp_conf_mean = (conf_accum / conf_count).tolist() if conf_count else [0]*17
    if debug and conf_count:
        print("\n===== 관절별 평균 신뢰도 (1.0=확실, 0=미검출) =====")
        for i, name in enumerate(KP_NAMES):
            bar = "#" * int(kp_conf_mean[i] * 20)
            warn = "  <-- 낮음!" if kp_conf_mean[i] < 0.5 else ""
            print(f"  {name:8s} {kp_conf_mean[i]:.2f} {bar}{warn}")
        # 무릎/발목 요약
        legs = [13, 14, 15, 16]
        leg_avg = np.mean([kp_conf_mean[i] for i in legs])
        upper = [5, 6, 7, 8, 11, 12]
        up_avg = np.mean([kp_conf_mean[i] for i in upper])
        print(f"\n  상체(어깨~골반) 평균: {up_avg:.2f}")
        print(f"  하체(무릎~발목) 평균: {leg_avg:.2f}")
        if leg_avg < 0.5 and up_avg >= 0.6:
            print("  => 무릎/발목만 신뢰도 낮음. 모델 키우기/imgsz↑/촬영각도 개선 필요.")
        elif leg_avg >= 0.6:
            print("  => 무릎/발목 신뢰도는 높음. 그래도 위치가 틀리면 '자신있게 틀리는' 경우.")
        print("=" * 50)


    # ── 3) One-Euro 스무딩 (시간 순서대로, 미검출은 건너뜀) ──
    oe = OneEuroFilter(OE_MIN_CUTOFF, OE_BETA, OE_DCUTOFF,
                       freq=30.0 / FRAME_STRIDE)
    for i in range(len(px_frames)):
        if px_frames[i] is None:
            continue
        sm = oe(px_frames[i])
        px_frames[i] = sm
        # 정규화본도 스무딩 결과 기반으로 갱신 (w,h는 sampled에서)
        _, w, h = sampled[i]
        norm_frames[i] = _normalize(sm, w, h)

    # ── 3.5) 스켈레톤 영상 저장 (옵션) ──
    skeleton_video = None
    if skeleton_out and sampled:
        _, w0, h0 = sampled[0]
        fps_out = max(30.0 / FRAME_STRIDE, 1.0)

        # 1차: 브라우저 호환 H.264(avc1) 시도, 실패 시 mp4v
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

        # 2차: mp4v로 저장됐으면 브라우저에서 안 보이므로 ffmpeg로 H.264 재인코딩
        skeleton_video = skeleton_out
        if used_codec == "mp4v":
            h264_path = os.path.splitext(skeleton_out)[0] + "_h264.mp4"
            if _reencode_h264(skeleton_out, h264_path):
                skeleton_video = h264_path

    # ── LSTM 분류: 유효 프레임(픽셀 좌표)을 34차원 시퀀스로 ──
    #   학습 시 정규화 없이 원본 픽셀 좌표를 사용했으므로 동일하게 픽셀 좌표 입력
    #   여러 시퀀스를 (N,30,34)로 쌓아 한 번에 추론 → GPU 오버헤드 감소
    valid_px = [f for f in px_frames if f is not None]
    seqs = []
    for i in range(0, len(valid_px) - SEQUENCE_LENGTH + 1, SEQ_STEP):
        chunk = valid_px[i:i + SEQUENCE_LENGTH]
        feat = np.array([c.flatten() for c in chunk], dtype=np.float32)  # (30,34)
        seqs.append(feat)

    if seqs:
        batch = np.stack(seqs, axis=0)                  # (N,30,34)
        with torch.no_grad():
            x = torch.tensor(batch, dtype=torch.float32).to(device)
            logits = lstm_model(x)                       # (N,4)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        seq_logits = list(probs)                         # 시퀀스별 확률
        mean_prob = np.mean(probs, axis=0)
        cls = int(np.argmax(mean_prob))
        exercise = CLASS_NAMES[cls]
        confidence = float(mean_prob[cls]) * 100
        n_sequences = len(seqs)
    else:
        # 시퀀스가 부족하면 분류 불가 → 기본값
        exercise = CLASS_NAMES[0]
        confidence = 0.0
        n_sequences = 0

    # ── 분류된 운동으로 프레임별 자세 평가 (정규화 좌표 사용) ──
    #   ok 여부뿐 아니라, 틀렸을 때의 구체적 설명(desc)도 함께 모은다.
    item_weight = {}                      # 항목별 가중치
    phase_count = {"down": 0, "up": 0, "none": 0}   # 단계 분포(진단용)
    phase_seq = []                        # 프레임 순서대로의 phase (횟수 카운트용)
    for norm in norm_frames:
        if norm is None:
            phase_seq.append("none")      # 미검출 프레임도 자리는 유지
            continue
        results, phase = evaluate_pose(exercise, norm)
        phase_count[phase] = phase_count.get(phase, 0) + 1
        phase_seq.append(phase)
        for r in results:
            item_ok.setdefault(r["label"], []).append(1 if r["ok"] else 0)
            item_weight[r["label"]] = r.get("weight", 1)
            if not r["ok"]:
                # 틀린 프레임의 설명 문구 수집 (pose_rules의 desc)
                item_bad_desc.setdefault(r["label"], []).append(r["desc"])

    # ── 항목별 점수(통과 비율) + 가중 평균 ──
    item_scores = []
    for label, oks in item_ok.items():
        ratio = sum(oks) / len(oks)
        item_scores.append({
            "label": label,
            "score": int(ratio * 100),
            "n": len(oks),
            "weight": item_weight.get(label, 1),
        })
    item_scores.sort(key=lambda x: x["score"])  # 낮은 점수 먼저

    # 전체 점수: 항목별 통과율을 가중 평균 (핵심 규칙 비중 ↑)
    if item_scores:
        tw = sum(s["weight"] for s in item_scores)
        avg_score = int(sum(s["score"] * s["weight"]
                            for s in item_scores) / tw) if tw else 0
    else:
        avg_score = 0

    # 단계 분포 진단 (운동 중 프레임이 너무 적으면 점수가 불안정)
    print(f"[진단] 단계 분포 - 운동중(down): {phase_count['down']}, "
          f"대기(up): {phase_count['up']}, 인식안됨(none): {phase_count['none']}")
    if phase_count["down"] < 5:
        print("  [경고] 운동 중으로 판정된 프레임이 매우 적습니다. "
              "phase 기준이 엄격하거나 운동 분류가 부정확할 수 있어요.")

    # ── 반복 횟수 / 유지 시간 카운트 ──
    #   분석에 쓰인 실효 FPS (FRAME_STRIDE 만큼 건너뛰어 추론했으므로 보정)
    eff_fps = max(30.0 / FRAME_STRIDE, 1.0)

    rep_count = None        # 반복형 운동: 횟수
    hold_seconds = None     # 플랭크: 유지 시간(초)

    if exercise == "플랭크":
        # 정적 운동 → "down"(자세 유지 중)으로 잡힌 프레임 수를 초로 환산
        hold_frames = phase_count.get("down", 0)
        hold_seconds = round(hold_frames / eff_fps, 1)
    else:
        # 반복형 운동 → phase가 up→down→up 한 사이클을 1회로 카운트.
        #   노이즈로 phase가 한두 프레임 튀는 것을 막기 위해 디바운스 적용:
        #   같은 상태가 MIN_HOLD 프레임 이상 연속될 때만 "확정 상태"로 전환.
        MIN_HOLD = max(int(eff_fps * 0.2), 2)   # 약 0.2초 이상 유지돼야 인정

        rep_count = 0
        stable_phase = "up"     # 현재 확정된 상태 (시작은 대기로 간주)
        run_phase = None        # 연속 관찰 중인 상태
        run_len = 0             # 그 상태가 연속된 프레임 수
        for ph in phase_seq:
            # none(미검출)은 직전 상태를 유지 (카운트에 영향 주지 않음)
            if ph == "none":
                run_phase = None
                run_len = 0
                continue
            if ph == run_phase:
                run_len += 1
            else:
                run_phase = ph
                run_len = 1
            # 같은 상태가 충분히 지속되면 확정 상태로 전환
            if run_len >= MIN_HOLD and ph != stable_phase:
                # up(대기) → down(운동 중)으로 확정 전환되는 순간 1회로 카운트
                if stable_phase == "up" and ph == "down":
                    rep_count += 1
                stable_phase = ph


    # ── 피드백 문구 ──
    #   항목별 점수에 따라 칭찬/주의 톤을 나누고,
    #   틀린 항목은 pose_rules의 구체적 교정 문구(desc)를 가장 많이 나온 것으로 첨부.
    def _most_common(lst):
        """리스트에서 가장 자주 나온 문구 반환 (없으면 None)"""
        if not lst:
            return None
        counts = {}
        for s in lst:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)

    feedbacks = []
    for s in item_scores:
        label = s["label"]
        score = s["score"]
        # 틀린 프레임에서 가장 자주 나온 구체적 교정 문구
        common_desc = _most_common(item_bad_desc.get(label, []))

        if score >= 80:
            title = "✅ " + label
            text = "대부분 잘 수행되고 있어요. 지금처럼 유지하세요!"
        elif score >= 50:
            title = "⚠️ " + label
            text = "조금만 더 다듬으면 될 것 같아요."
            if common_desc:
                # 괄호 안 각도/수치 부분은 떼고 핵심 교정 문구만 사용
                text += f" {common_desc.split('(')[0].strip()}"
        else:
            title = "❌ " + label
            text = "이 부분은 교정이 필요해요."
            if common_desc:
                text += f" {common_desc.split('(')[0].strip()}"

        feedbacks.append((title, text))

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