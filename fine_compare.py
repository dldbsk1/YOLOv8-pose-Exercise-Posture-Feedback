# -*- coding: utf-8 -*-
"""
파인튜닝 모델 vs 사전학습 모델 비교

"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from config import BASE_DIR, YOLO_POSE_PT

# ===================== 설정 =====================
VIDEO = BASE_DIR / "videos" / "test_video_2.mov"  
PRETRAINED = "yolov8s-pose.pt"                    
FINETUNED = YOLO_POSE_PT                          
OUT = BASE_DIR / "outputs" / "compare_pretrained_vs_finetuned_ph2.mp4"

IMGSZ = 960
CONF = 0.25
KP_CONF = 0.5
FRAME_STRIDE = 2
# ===============================================

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


def pick_main(res):
    if res.keypoints is None or res.keypoints.xy is None:
        return None, None
    xy = res.keypoints.xy.cpu().numpy()
    if xy.shape[0] == 0:
        return None, None
    conf = (res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones((xy.shape[0], xy.shape[1]), np.float32))
    idx = 0
    if xy.shape[0] > 1 and res.boxes is not None:
        areas = [b[2]*b[3] for b in res.boxes.xywh.cpu().numpy()]
        idx = int(np.argmax(areas))
    return xy[idx], conf[idx]


def draw(frame, xy, conf, color_line, color_pt):
    if xy is None:
        return
    for i, j in SKELETON:
        if conf[i] >= KP_CONF and conf[j] >= KP_CONF:
            cv2.line(frame, tuple(xy[i].astype(int)),
                     tuple(xy[j].astype(int)), color_line, 2)
    for k, (x, y) in enumerate(xy):
        if conf[k] >= KP_CONF:
            cv2.circle(frame, (int(x), int(y)), 4, color_pt, -1)


def main():
    m_pre = YOLO(PRETRAINED)
    m_fin = YOLO(FINETUNED)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("영상 못 엶:", VIDEO); return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30) / FRAME_STRIDE
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w*2, h))

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % FRAME_STRIDE != 0:
            fidx += 1; continue
        fidx += 1

        left = frame.copy()
        right = frame.copy()

        r_pre = m_pre.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        xy, cf = pick_main(r_pre)
        draw(left, xy, cf, (200,200,200), (0,200,255))
        cv2.putText(left, "PRETRAINED", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        r_fin = m_fin.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        xy2, cf2 = pick_main(r_fin)
        draw(right, xy2, cf2, (200,200,200), (255,200,0))
        cv2.putText(right, "FINETUNED", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        combined = np.hstack([left, right])
        writer.write(combined)

    cap.release()
    writer.release()
    print(f"비교 영상 저장: {OUT}")
    print("좌(사전학습) vs 우(파인튜닝)")


if __name__ == "__main__":
    main()