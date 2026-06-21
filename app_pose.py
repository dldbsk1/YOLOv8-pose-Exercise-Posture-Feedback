# -*- coding: utf-8 -*-
import os
import tempfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
import torch

from PIL import ImageFont, ImageDraw, Image
from analyzer import analyze_video, load_models, CLASS_NAMES, SEQUENCE_LENGTH
from analyzer import _pick_main, _mask_low_conf, _normalize, IMGSZ, DET_CONF, KP_CONF, draw_skeleton_on
from pose_rules import evaluate_pose, calc_score, PHASE_LABEL

st.set_page_config(page_title="운동 자세 분석", layout="wide")

# --------------------
# CSS
# --------------------
st.markdown("""
<style>
.main { background-color: #f7f8fa; }
.block-container{ padding-top: 2rem; max-width: 1200px; }
.card{
    background:white; padding:25px; border-radius:20px;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}
.result-title{ color:#22c55e; font-size:42px; font-weight:700; }
.tip-card{
    background:#f8fafc; border:1px solid #e5e7eb; border-radius:15px;
    padding:20px; margin-bottom:15px;
}

/* ===== 항목별 점수 막대 (80% 기준 초록/노랑) ===== */
.score-item { margin-bottom:18px; }
.score-name { font-size:15px; font-weight:700; color:#1a1a1a; margin:0 0 6px 0; }
.bar-row    { display:flex; align-items:center; gap:12px; }
.bar-bg     { flex:1; background:#e2e8f0; border-radius:999px; height:8px; overflow:hidden; }
.bar-green  { background:#22c55e; height:100%; border-radius:999px; }
.bar-yellow { background:#facc15; height:100%; border-radius:999px; }
.pct-green  { font-size:14px; font-weight:700; color:#22c55e; min-width:40px; text-align:right; }
.pct-yellow { font-size:14px; font-weight:700; color:#ca8a04; min-width:40px; text-align:right; }

/* ===== AI 피드백 박스 초록 계열 ===== */
.fb-card {
    background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px;
    padding:14px 18px; margin-bottom:12px;
}
.fb-title { font-size:14px; font-weight:700; color:#16a34a; margin:0 0 4px 0; }
.fb-text  { font-size:13px; color:#64748b; margin:0; }

/* ===== AI 피드백 박스 노랑 계열 (80% 미만 항목) ===== */
.fb-card-yellow {
    background:#fefce8; border:1px solid #fde68a; border-radius:12px;
    padding:14px 18px; margin-bottom:12px;
}
.fb-title-yellow { font-size:14px; font-weight:700; color:#ca8a04; margin:0 0 4px 0; }

/* ===== 분석 시작(primary) 버튼 초록색 ===== */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #22c55e !important;
    border-color: #22c55e !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #16a34a !important;
    border-color: #16a34a !important;
    color: white !important;
}
.stButton > button[kind="primary"]:active,
.stButton > button[kind="primary"]:focus {
    background-color: #16a34a !important;
    border-color: #16a34a !important;
    color: white !important;
    box-shadow: none !important;
}

/* ===== 로딩 화면 전용 스타일 ===== */
.loading-wrap { background:white; border-radius:16px; padding:36px 28px;
                margin-bottom:20px; box-shadow:0 1px 6px rgba(0,0,0,0.07); }
.loading-title { font-size:28px; font-weight:800; color:#1a1a1a; margin:0 0 6px 0; }
.loading-sub   { font-size:14px; color:#94a3b8; margin:0 0 28px 0; }
.step-row { display:flex; align-items:flex-start; gap:14px; margin-bottom:18px; }
.step-icon-done   { width:36px; height:36px; min-width:36px; background:#22c55e; border-radius:50%;
                    display:flex; align-items:center; justify-content:center; color:white; font-size:16px; font-weight:800; }
.step-icon-active { width:36px; height:36px; min-width:36px; background:#dcfce7; border:2px solid #22c55e; border-radius:50%;
                    display:flex; align-items:center; justify-content:center; font-size:18px; }
.step-icon-wait   { width:36px; height:36px; min-width:36px; background:#f1f5f9; border-radius:50%;
                    display:flex; align-items:center; justify-content:center; font-size:16px; }
.step-text-done   { font-size:14px; font-weight:700; color:#22c55e; margin:0 0 2px 0; }
.step-text-active { font-size:14px; font-weight:700; color:#1a1a1a; margin:0 0 2px 0; }
.step-text-wait   { font-size:14px; font-weight:700; color:#94a3b8; margin:0 0 2px 0; }
.step-sub { font-size:12px; color:#94a3b8; margin:0; }
</style>
""", unsafe_allow_html=True)


# --------------------
# 사이드바: 검출 임계값 슬라이더
# --------------------
# 0.1 쪽으로 갈수록 민감하게 → 더 많이 검출 (오탐 가능성 ↑)
# 0.5 기본값 → 균형 잡힌 검출
# 1.0 쪽으로 갈수록 엄격하게 → 확실한 것만 검출 (누락 가능성 ↑)
with st.sidebar:
    st.markdown("#### 🎯 검출 임계값")
    conf_threshold = st.slider(
        "임계값",
        min_value=0.1, max_value=1.0,
        value=0.5, step=0.1,
        label_visibility="collapsed"
    )
    st.markdown(
        f"<p style='font-size:12px; color:#94a3b8; margin:4px 0 16px 0;'>"
        f"현재 임계값: <b>{conf_threshold}</b></p>",
        unsafe_allow_html=True
    )
    st.markdown("---")


# --------------------
# 로딩 화면 구성 요소
# --------------------
# 5단계 정의: (아이콘 이모지, 제목)
# 실제 analyzer 흐름에 맞춤:
#   영상 수신 → YOLO 관절 추출 → LSTM 운동 분류 → 규칙 기반 자세 분석 → 결과 생성
STEPS = [
    ("📥", "1. 영상 수신"),
    ("🦴", "2. 관절 추출"),
    ("🏃", "3. 운동 분류"),
    ("📊", "4. 자세 분석"),
    ("📋", "5. 결과 생성"),
]


def render_steps_html(current):
    """
    단계 카드 HTML 생성.
    current: 현재 진행 중인 단계 인덱스(0~4). current보다 앞 단계는 완료(✓),
             current는 진행 중, 뒤는 대기.
    """
    rows = ""
    for i, (emoji, title) in enumerate(STEPS):
        if i < current:
            # 완료
            rows += (f"<div class='step-row'><div class='step-icon-done'>✓</div>"
                     f"<div><p class='step-text-done'>{title}</p>"
                     f"<p class='step-sub'>완료</p></div></div>")
        elif i == current:
            # 진행 중
            rows += (f"<div class='step-row'><div class='step-icon-active'>{emoji}</div>"
                     f"<div><p class='step-text-active'>{title}</p>"
                     f"<p class='step-sub'>진행 중</p></div></div>")
        else:
            # 대기
            rows += (f"<div class='step-row'><div class='step-icon-wait'>{emoji}</div>"
                     f"<div><p class='step-text-wait'>{title}</p>"
                     f"<p class='step-sub'>대기 중</p></div></div>")
    return f"""
    <div class='card'>
        <p style='font-size:15px; font-weight:700; margin:0 0 16px 0;'>분석 단계</p>
        {rows}
    </div>
    """


def render_fixed_parts():
    """단계 카드를 제외한 고정 영역 (가운데/오른쪽). 분석 내내 변하지 않음."""
    st.markdown("""
    <div class='loading-wrap'>
        <p class='loading-title'>운동 자세 분석 중입니다 ✨</p>
        <p class='loading-sub'>잠시만 기다려주세요! 정확한 분석을 위해 영상을 꼼꼼히 확인하고 있어요.</p>
    </div>
    """, unsafe_allow_html=True)


def render_loading(loading_slot, current_step):
    """로딩 화면 전체(타이틀 + 3컬럼)를 한 번에 그린다.
    container 안에 empty를 중첩하지 않으므로 streamlit 1.58 렌더링 충돌(setIn)이 없음.
    콜백마다 이 함수를 호출해 화면을 통째로 다시 그린다(단계 카드만 current_step에 따라 바뀜)."""
    with loading_slot.container():
        # 상단 타이틀 (고정)
        render_fixed_parts()

        col_steps, col_mid, col_right = st.columns([1, 1.4, 1])

        # 왼쪽: 단계 카드 (current_step에 따라 갱신)
        with col_steps:
            st.markdown(render_steps_html(current_step), unsafe_allow_html=True)

        # 가운데: 고정
        with col_mid:
            st.markdown("""
            <div class='card' style='text-align:center;'>
                <p style='font-size:13px; color:#94a3b8; margin:0 0 6px 0;'>분석 중인 동작</p>
                <p style='font-size:28px; font-weight:800; color:#22c55e; margin:0 0 16px 0;'>분석 중...</p>
                <p style='font-size:60px; margin:0 0 16px 0;'>🏋️</p>
                <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px; padding:14px 16px; text-align:left;'>
                    <p style='font-size:13px; color:#16a34a; font-weight:700; margin:0 0 4px 0;'>🤖 AI가 영상 속 움직임을 분석하고 있어요.</p>
                    <p style='font-size:13px; color:#64748b; margin:0;'>조금만 기다려주시면 정확한 결과를 확인할 수 있어요!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 오른쪽: 고정
        with col_right:
            st.markdown("""
            <div class='card'>
                <p style='font-size:15px; font-weight:700; margin:0 0 16px 0;'>분석이 완료되면?</p>
                <div class='step-row'>
                    <div class='step-icon-done' style='background:#dcfce7;'><span style='color:#22c55e;'>✓</span></div>
                    <div><p style='font-size:14px; font-weight:700; color:#1a1a1a; margin:0 0 2px 0;'>자세 점수 확인</p>
                         <p style='font-size:12px; color:#94a3b8; margin:0;'>100점 만점으로 자세를 평가해드려요.</p></div>
                </div>
                <div class='step-row'>
                    <div class='step-icon-done' style='background:#dcfce7;'><span style='color:#22c55e;'>✓</span></div>
                    <div><p style='font-size:14px; font-weight:700; color:#1a1a1a; margin:0 0 2px 0;'>부위별 자세 피드백</p>
                         <p style='font-size:12px; color:#94a3b8; margin:0;'>어떤 부분을 개선하면 좋을지 알려드려요.</p></div>
                </div>
                <div class='step-row'>
                    <div class='step-icon-done' style='background:#dcfce7;'><span style='color:#22c55e;'>✓</span></div>
                    <div><p style='font-size:14px; font-weight:700; color:#1a1a1a; margin:0 0 2px 0;'>자세 정확도 제공</p>
                         <p style='font-size:12px; color:#94a3b8; margin:0;'>전체 자세 정확도를 퍼센트(%)로 보여드려요.</p></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def progress_to_step(p):
    """진행률(0.0~1.0) → 단계 인덱스(0~4) 매핑.
    analyzer는 추론을 0~0.85로, 마무리를 1.0으로 보냄.
    0~0.85 구간을 1~3단계(자세감지/관절추적), 0.85~1.0을 4~5단계로 나눔."""
    if p < 0.05:
        return 0   # 영상 수신
    elif p < 0.45:
        return 1   # 자세 감지
    elif p < 0.85:
        return 2   # 관절 추적
    elif p < 1.0:
        return 3   # 자세 분석
    else:
        return 4   # 결과 생성

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT = ImageFont.truetype(FONT_PATH, 24)

def draw_korean_text(img, text, pos,
                     color=(0, 200, 0)):

    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    draw.text(
        pos,
        text,
        font=FONT,
        fill=color
    )

    return np.array(img_pil)
# --------------------
# 입력 방식 선택
# --------------------
st.markdown("## 운동 자세 분석")
mode = st.radio("입력 방식", ["영상 업로드", "실시간 웹캠"], horizontal=True)

# 세션 상태
if "result" not in st.session_state:
    st.session_state["result"] = None


# ====================================================
# 1) 영상 업로드 모드
# ====================================================
if mode == "영상 업로드":
    uploaded = st.file_uploader("운동 영상을 올려주세요",
                                type=["mp4", "mov", "avi", "m4v"])

    if uploaded is not None:
        # 임시 파일로 저장
        suffix = os.path.splitext(uploaded.name)[1]
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tfile.write(uploaded.read())
        tfile.flush()
        video_path = tfile.name

        if st.button("분석 시작", type="primary"):
            # 로딩 화면 전체를 담을 단일 슬롯 (container 안에 empty 중첩 X)
            loading = st.empty()
            # 첫 화면 그리기 (0단계)
            render_loading(loading, 0)

            # 진행률 콜백: 로딩 화면 전체를 현재 단계로 다시 그림
            def cb(p):
                render_loading(loading, progress_to_step(p))

            # 실제 분석 수행
            skel_path = os.path.splitext(video_path)[0] + "_skeleton.mp4"
            try:
                st.session_state["result"] = analyze_video(
                    video_path, progress_cb=cb, skeleton_out=skel_path,
                    debug=True, det_conf=conf_threshold)
                st.session_state["video_path"] = video_path
                # 분석 끝 → 마지막 단계(결과 생성)까지 표시
                render_loading(loading, 4)
            except Exception as e:
                st.session_state["result"] = None
                loading.empty()
                st.error(f"분석 실패: {e}")

            # 로딩 화면 지우고 결과로
            loading.empty()
            if st.session_state.get("result"):
                st.rerun()

    # 결과 표시
    res = st.session_state["result"]
    if res:
        # ── 상단: 영상(스켈레톤) + 분류 결과 ──
        left, right = st.columns([1, 1])
        with left:
            skel = res.get("skeleton_video")
            if skel and os.path.exists(skel):
                with open(skel, "rb") as vf:
                    st.video(vf.read())
                st.caption("관절 스켈레톤이 표시된 분석 영상")
            else:
                st.video(st.session_state.get("video_path"))
        with right:
            # 반복 횟수(반복형) 또는 유지 시간(플랭크) 표시 문구
            rep = res.get("rep_count")
            hold = res.get("hold_seconds")
            if hold is not None:
                count_html = (f"<p style='color:gray; margin:8px 0 0 0;'>유지 시간</p>"
                              f"<div class='result-title'>{hold}초</div>")
            elif rep is not None:
                count_html = (f"<p style='color:gray; margin:8px 0 0 0;'>반복 횟수</p>"
                              f"<div class='result-title'>{rep}회</div>")
            else:
                count_html = ""

            st.markdown(f"""
            <div class="card">
            <p style="color:gray;">예측 운동</p>
            <div class="result-title">{res['exercise']}</div>
            <br>
            <h1>{res['confidence']}%</h1>
            {count_html}
            <p>사용된 시퀀스 수 : {res['n_sequences']}개</p>
            <p>유효 관절 프레임 : {res['valid_frames']}개</p>
            </div>
            """, unsafe_allow_html=True)

            # 무릎/발목 신뢰도 진단
            kc = res.get("kp_conf_mean")
            if kc:
                leg_idx = {"왼무릎": 13, "오무릎": 14, "왼발목": 15, "오발목": 16}
                leg_avg = sum(kc[i] for i in leg_idx.values()) / 4
                with st.expander(f"관절 신뢰도 진단 (하체 평균 {leg_avg:.2f})"):
                    for name, i in leg_idx.items():
                        st.write(f"{name}: {kc[i]:.2f}")
                        st.progress(min(kc[i], 1.0))
                    if leg_avg < 0.5:
                        st.warning("무릎/발목 신뢰도 낮음 → 모델 키우기(m), imgsz↑, 촬영각도 개선 권장")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ── 자세 분석 결과 ──
        st.markdown("## 자세 분석 결과")
        c1, c2 = st.columns([1, 2])
        with c1:
            avg = res["avg_score"]
            grade = "GOOD!" if avg >= 80 else "OK" if avg >= 50 else "NEEDS WORK"
            color = "#22c55e" if avg >= 80 else "#eab308" if avg >= 50 else "#ef4444"
            msg = ("전반적으로 좋은 자세예요 👏" if avg >= 80
                   else "조금만 더 다듬으면 좋아요" if avg >= 50
                   else "자세 교정이 필요해요")
            st.markdown(f"""
            <div class="card" style="text-align:center;">
            <h1 style="font-size:60px;color:{color};">{avg}%</h1>
            <h3>{grade}</h3>
            <p>{msg}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("### 항목별 점수")
            for item in res["item_scores"]:
                score = item["score"]
                good = score >= 80           # 80% 이상이면 초록, 미만이면 노랑
                bar_cls = "bar-green" if good else "bar-yellow"
                pct_cls = "pct-green" if good else "pct-yellow"
                st.markdown(f"""
                <div class='score-item'>
                  <p class='score-name'>{item['label']}</p>
                  <div class='bar-row'>
                    <div class='bar-bg'><div class='{bar_cls}' style='width:{score}%;'></div></div>
                    <span class='{pct_cls}'>{score}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── AI 피드백 ──
        st.markdown("## AI 자세 피드백")
        # 항목 이름 → 점수 매핑 (피드백 색을 항목별 점수에 맞추기 위함)
        score_by_label = {it["label"]: it["score"] for it in res["item_scores"]}
        for title, text in res["feedbacks"]:
            # 제목(title)에 항목 이름(label)이 포함돼 있으므로, 매칭되는 점수를 찾음
            matched_score = None
            for label, sc in score_by_label.items():
                if label in title:
                    matched_score = sc
                    break
            # 80% 이상이면 초록, 미만이면 노랑 (항목별 점수 막대와 동일 규칙)
            good = matched_score is None or matched_score >= 80
            card_cls = "fb-card" if good else "fb-card-yellow"
            title_cls = "fb-title" if good else "fb-title-yellow"
            st.markdown(f"""
            <div class='{card_cls}'>
              <p class='{title_cls}'>{title}</p>
              <p class='fb-text'>{text}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── 팁 (점수 낮은 항목 위주) ──
        st.markdown("## 분석 팁")
        low = [s for s in res["item_scores"] if s["score"] < 80][:3]
        if low:
            cols = st.columns(len(low))
            for col, s in zip(cols, low):
                with col:
                    st.markdown(f"""
                    <div class="tip-card">
                    ### {s['label']}
                    이 항목의 통과율이 {s['score']}% 예요. 집중해서 교정해보세요.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("모든 항목이 80% 이상이에요! 아주 좋아요 👏")

        if st.button("분석 다시 하기"):
            st.session_state["result"] = None
            st.rerun()


# ====================================================
# 2) 실시간 웹캠 모드
# ====================================================
else:
    st.markdown("웹캠으로 실시간 관절을 추출하고, LSTM이 운동 종류를 자동 분류한 뒤 해당 운동 규칙으로 자세를 평가합니다.")

    run_cam = st.checkbox("웹캠 시작")
    frame_slot = st.empty()
    info_slot = st.empty()

    # LSTM 분류 안정화용 설정
    MIN_CLASSIFY_FRAMES = SEQUENCE_LENGTH     # 최소 30프레임 모이면 첫 분류
    SMOOTH_PRED_WINDOW = 8                    # 최근 예측 8개를 평균내서 흔들림 완화
    CLASSIFY_EVERY_N_FRAMES = 3               # 매 프레임 분류하지 않고 3프레임마다 분류

    if run_cam:
        pose_model, lstm_model, device = load_models()
        cap = cv2.VideoCapture(0)

        # 최근 관절 시퀀스 저장: LSTM은 analyzer.py와 동일하게 픽셀 좌표 34차원을 입력으로 사용
        seq_buffer = deque(maxlen=SEQUENCE_LENGTH)
        pred_buffer = deque(maxlen=SMOOTH_PRED_WINDOW)
        auto_exercise = None
        auto_confidence = 0.0
        frame_count = 0

        if not cap.isOpened():
            st.error("웹캠을 열 수 없습니다.")
        else:
            while run_cam:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                h, w = frame.shape[:2]
                res = pose_model.predict(frame, conf=conf_threshold,
                                         imgsz=IMGSZ, verbose=False)[0]
                xy, conf = _pick_main(res)

                if xy is not None:
                    masked = _mask_low_conf(xy, conf, KP_CONF).astype(np.float32)
                    norm = _normalize(masked, w, h)

                    # 스켈레톤 그리기 (관절 추적 확인용)
                    try:
                        draw_skeleton_on(frame, masked)
                    except Exception:
                        pass

                    # LSTM 입력 버퍼 업데이트: (17,2) → 34차원
                    seq_buffer.append(masked.flatten())

                    # 30프레임 이상 모이면 자동 운동 분류
                    if (len(seq_buffer) >= MIN_CLASSIFY_FRAMES
                            and frame_count % CLASSIFY_EVERY_N_FRAMES == 0):
                        seq = np.array(seq_buffer, dtype=np.float32)  # (30,34)
                        x = torch.tensor(seq[None, :, :], dtype=torch.float32).to(device)
                        with torch.no_grad():
                            logits = lstm_model(x)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_buffer.append(probs)

                        mean_prob = np.mean(np.stack(pred_buffer, axis=0), axis=0)
                        cls = int(np.argmax(mean_prob))
                        auto_exercise = CLASS_NAMES[cls]
                        auto_confidence = float(mean_prob[cls]) * 100

                    # 아직 LSTM 시퀀스가 부족하면 자세 평가는 대기
                    if auto_exercise is None:
                        need = MIN_CLASSIFY_FRAMES - len(seq_buffer)
                        label = f"운동 분류 준비 중... {len(seq_buffer)}/{MIN_CLASSIFY_FRAMES}"
                        frame = draw_korean_text(frame, label, (10, 10), color=(220, 150, 0))
                        info_slot.markdown(
                            f"_LSTM 분류를 위해 프레임을 모으는 중입니다. 약 {max(need, 0)}프레임 남음._")
                    else:
                        # 자동 분류된 운동명으로 규칙 기반 자세 평가
                        results, phase = evaluate_pose(auto_exercise, norm,
                                                       only_active=False)
                        score = calc_score(results)
                        label = (f"[{auto_exercise} {auto_confidence:.1f}%] "
                                 f"{PHASE_LABEL.get(phase, phase)}")
                        if results:
                            label += f"  score:{score}"
                        frame = draw_korean_text(frame, label, (10, 10))

                        if results:
                            info_lines = [
                                f"**자동 분류:** {auto_exercise} ({auto_confidence:.1f}%)",
                                f"**현재 단계:** {PHASE_LABEL.get(phase, phase)}",
                                f"**현재 점수:** {score}",
                                "---",
                            ]
                            info_lines += [f"{'✅' if r['ok'] else '❌'} {r['label']}"
                                           for r in results]
                            info_slot.markdown("  \n".join(info_lines))
                        else:
                            info_slot.markdown(
                                f"**자동 분류:** {auto_exercise} ({auto_confidence:.1f}%)  \n"
                                f"_자세를 인식 중입니다 (단계: {PHASE_LABEL.get(phase, phase)})_")
                else:
                    # 사람이 안 잡히면 시퀀스를 초기화해서 이전 운동 분류가 계속 남지 않게 함
                    seq_buffer.clear()
                    pred_buffer.clear()
                    auto_exercise = None
                    auto_confidence = 0.0
                    cv2.putText(frame, "No person detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                    info_slot.markdown("_사람이 화면에 보이도록 위치를 조정하세요_")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, channels="RGB",
                                 use_container_width=True)
            cap.release()
    else:
        st.info("'웹캠 시작'을 체크하면 실시간 자동 운동 분류 + 자세 평가가 시작됩니다.")