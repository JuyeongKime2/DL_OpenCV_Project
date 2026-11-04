
"""실시간 산업현장 작업자 수신호 인식 Streamlit 앱."""

from collections import Counter, deque

import time

import cv2
import numpy as np
import streamlit as st
import torch

from config import (
    CLASS_KOREAN,
    EXTRACTORS,
    MIN_CONSISTENT_VOTES,
    MIN_MODEL_SCORE,
    MODEL_DIR,
    PREDICTION_INTERVAL,
    SEQUENCE_LENGTH,
    SMOOTHING_WINDOW,
)

from pose_utils import (
    get_extractor,
    has_valid_detection,
)

from train import GestureNet


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(

    page_title="산업현장 수신호 인식 AI",

    page_icon="🏗️",

    layout="wide"
)

st.title(
    "🏗️ AI 산업현장 작업자 수신호 인식"
)

st.caption(
    "세션 독립 평가 모델 · 4개 클래스 분류 · 연속 예측 안정화"
)

st.sidebar.warning(
    "시연 및 연구용 프로토타입입니다. "
    "실제 산업 장비 제어나 안전 시스템에 직접 연결하지 마세요."
)

st.sidebar.info(

    "관절 미검출 프레임은 분류하지 않습니다. "

    "모델 점수는 softmax 출력이며 "
    "보정된 실제 확률을 의미하지 않습니다."
)

st.sidebar.caption(

    f"표시 기준: 모델 점수 {MIN_MODEL_SCORE:.2f} 이상, "

    f"최근 {SMOOTHING_WINDOW}회 예측 중 "

    f"{MIN_CONSISTENT_VOTES}회 이상 일치"
)


# ============================================================
# 2. 학습된 모델 확인
# ============================================================

available_extractors = [

    name

    for name in EXTRACTORS

    if (
        MODEL_DIR
        / f"{name}_lstm_best.pth"
    ).exists()
]

if not available_extractors:

    st.error(

        "학습된 모델 파일을 찾지 못했습니다.\n\n"

        "먼저 데이터 수집 후 "
        "`python train.py --extractor yolo`를 실행하세요."
    )

    st.stop()


recommended_extractor = available_extractors[0]

best_extractor_file = (
    MODEL_DIR / "best_extractor.txt"
)

if best_extractor_file.exists():

    candidate = best_extractor_file.read_text(
        encoding="utf-8"
    ).strip()

    if candidate in available_extractors:

        recommended_extractor = candidate


selected_extractor = st.sidebar.selectbox(

    "사용할 관절 추출 모델",

    options=available_extractors,

    index=available_extractors.index(
        recommended_extractor
    ),

    format_func=lambda value: EXTRACTORS[value]["name"]
)


# ============================================================
# 3. 모델 및 특징 추출기 로드
# ============================================================

@st.cache_resource
def load_model_and_extractor(extractor_name):

    model_path = (
        MODEL_DIR
        / f"{extractor_name}_lstm_best.pth"
    )

    model = GestureNet(
        EXTRACTORS[extractor_name]["num_features"]
    )

    try:

        state_dict = torch.load(

            model_path,

            map_location="cpu",

            weights_only=True
        )

    except TypeError:

        # 구버전 PyTorch 호환
        state_dict = torch.load(
            model_path,
            map_location="cpu"
        )

    model.load_state_dict(state_dict)

    model.eval()

    extractor = get_extractor(
        extractor_name
    )

    return model, extractor


try:

    model, extractor = load_model_and_extractor(
        selected_extractor
    )

except Exception as exc:

    st.error(
        f"모델 또는 특징 추출기를 불러오지 못했습니다: {exc}"
    )

    st.stop()


# ============================================================
# 4. 실시간 웹캠 인식
# ============================================================

st.markdown(
    f"### 선택 모델: {EXTRACTORS[selected_extractor]['name']}"
)

run_camera = st.checkbox(
    "웹캠 시작",
    key="run_camera"
)

video_area = st.empty()

status_area = st.empty()


if run_camera:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        st.error(
            "웹캠을 열 수 없습니다. "
            "카메라 연결과 사용 권한을 확인하세요."
        )

        st.stop()

    feature_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    current_label = "수신호 대기"

    current_score = 0.0

    frame_count = 0

    try:

        while run_camera:

            ok, frame = camera.read()

            if not ok:

                st.error(
                    "웹캠 프레임을 읽을 수 없습니다."
                )

                break

            frame = cv2.flip(frame, 1)

            feature, annotated = extractor.extract(
                frame
            )

            feature = np.asarray(
                feature,
                dtype=np.float32
            )

            feature_ok = (

                feature.size
                == EXTRACTORS[selected_extractor]["num_features"]

                and np.isfinite(feature).all()

                and has_valid_detection(
                    feature,
                    selected_extractor
                )
            )

            # 관절이 검출되지 않으면 예측하지 않음
            if not feature_ok:

                feature_buffer.clear()

                prediction_history.clear()

                current_label = "수신호 대기"

                current_score = 0.0

            else:

                feature_buffer.append(
                    feature
                )

            frame_count += 1

            # 30프레임 확보 후 5프레임마다 추론
            if (

                feature_ok

                and len(feature_buffer) == SEQUENCE_LENGTH

                and frame_count % PREDICTION_INTERVAL == 0
            ):

                sequence = np.stack(
                    feature_buffer
                ).astype(np.float32)

                input_tensor = torch.from_numpy(
                    sequence
                ).unsqueeze(0)

                with torch.no_grad():

                    logits = model(
                        input_tensor
                    )

                    scores = torch.softmax(
                        logits,
                        dim=1
                    )[0].cpu().numpy()

                predicted_class = int(
                    np.argmax(scores)
                )

                model_score = float(
                    scores[predicted_class]
                )

                prediction_history.append(
                    (predicted_class, model_score)
                )

                # 최근 예측 결과의 투표
                vote_counts = Counter(

                    label

                    for label, _ in prediction_history
                )

                most_common_class, vote_count = (
                    vote_counts.most_common(1)[0]
                )

                matched_scores = [

                    score

                    for label, score in prediction_history

                    if label == most_common_class
                ]

                average_score = (

                    float(np.mean(matched_scores))

                    if matched_scores

                    else 0.0
                )

                # 최종 판정
                if (

                    len(prediction_history)
                    >= MIN_CONSISTENT_VOTES

                    and vote_count
                    >= MIN_CONSISTENT_VOTES

                    and average_score
                    >= MIN_MODEL_SCORE
                ):

                    current_label = CLASS_KOREAN[
                        most_common_class
                    ]

                    current_score = average_score

                elif model_score < MIN_MODEL_SCORE:

                    current_label = "판단 보류"

                    current_score = model_score

                else:

                    current_label = "연속 동작 확인 중"

                    current_score = model_score

            # 결과 화면
            display_color = (

                (0, 180, 0)

                if current_score >= MIN_MODEL_SCORE

                else (0, 165, 255)
            )

            cv2.putText(

                annotated,

                f"{current_label} | score: {current_score:.2f}",

                (12, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.70,

                display_color,

                2
            )

            cv2.putText(

                annotated,

                f"Buffer: {len(feature_buffer)}/{SEQUENCE_LENGTH}",

                (12, 65),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 0),

                1
            )

            video_area.image(

                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),

                channels="RGB",

                use_container_width=True
            )

            status_area.metric(

                "실시간 인식 결과",

                current_label,

                f"모델 점수 {current_score:.2f} (보정 확률 아님)"
            )

            time.sleep(0.015)

    finally:

        camera.release()


else:

    st.info(

        "웹캠 시작을 체크하면 실시간 인식을 진행합니다. "

        "동작 전환 직후에는 30프레임이 쌓일 때까지 기다려 주세요."
    )


# ============================================================
# 5. 안내
# ============================================================

st.markdown("---")

st.markdown("**인식 클래스**")

st.write(

    "STOP: 작업 정지 · "

    "UP: 장비 상승 · "

    "DOWN: 장비 하강 · "

    "OTHER: 사람이 검출되지만 지정 수신호가 아닌 상태"
)

st.caption(

    "사람/손 관절 자체가 검출되지 않으면 "
    "OTHER로 분류하지 않고 '수신호 대기'로 표시합니다. "

    "실제 오탐률은 학습에 사용하지 않은 별도 테스트 영상으로 확인해야 합니다."
)
