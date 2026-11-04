"""
실시간 수신호 인식 Streamlit 앱.
사이드바에서 추출기(YOLOv8-Pose / Hands / Holistic)를 선택하여
해당 모델로 실시간 추론합니다.
"""
import cv2
import streamlit as st
import numpy as np
import torch
from collections import deque

from pose_utils import get_extractor
from config import MODEL_DIR, CLASS_NAMES, CLASS_KOREAN, SEQUENCE_LENGTH, EXTRACTORS
from train import GestureNet

st.set_page_config(page_title="산업현장 수신호 인식 AI", page_icon="🏗️", layout="wide")
st.title("🏗️ AI 산업현장 작업자 수신호 인식")
st.caption("YOLOv8-Pose / MediaPipe Hands / Holistic + LSTM 비교")

# --- 사이드바 ---
st.sidebar.header("모델 선택")
extractor_choice = st.sidebar.selectbox(
    "관절 추출기",
    options=list(EXTRACTORS.keys()),
    format_func=lambda x: EXTRACTORS[x]["name"],
)
st.sidebar.info(EXTRACTORS[extractor_choice]["description"])
st.sidebar.warning("시연용 프로토타입입니다. 실제 산업장비 제어에 사용하지 마세요.")

@st.cache_resource
def load_model(extractor_name: str):
    num_features = EXTRACTORS[extractor_name]["num_features"]
    weight_path = MODEL_DIR / f"{extractor_name}_lstm_best.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"{extractor_name} 모델이 없습니다. 먼저 python train.py를 실행하세요.")
    model = GestureNet(num_features)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    extractor = get_extractor(extractor_name)
    return extractor, model, num_features

try:
    extractor, model, num_features = load_model(extractor_choice)
except Exception as e:
    st.error(str(e))
    st.stop()

run = st.checkbox("웹캠 시작", key="run")
frame_area = st.empty()
result_area = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        st.stop()

    buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_label = "대기 중"
    last_conf = 0.0

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("프레임을 읽을 수 없습니다.")
            break
        frame = cv2.flip(frame, 1)
        feature, annotated = extractor.extract(frame)

        if len(feature) != num_features:
            feature = np.zeros(num_features, dtype=np.float32)
        buffer.append(feature)

        if len(buffer) == SEQUENCE_LENGTH:
            seq = np.stack(buffer).astype(np.float32)
            with torch.no_grad():
                logits = model(torch.tensor(seq).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0].numpy()
                pred = int(probs.argmax())
                last_label = CLASS_KOREAN[pred]
                last_conf = float(probs[pred])

        # 화면에 결과 표시
        color = (0, 255, 0) if last_conf > 0.6 else (0, 165, 255)
        cv2.putText(annotated, f"{last_label} ({last_conf:.2f})",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(annotated, EXTRACTORS[extractor_choice]["name"],
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        frame_area.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        result_area.metric("인식 결과", last_label, f"신뢰도 {last_conf:.2f}")

    cap.release()

st.markdown("---")
st.caption("데이터 수집 → `python collect_data.py --extractor <yolo|hands|holistic>` → `python train.py` → 이 앱 실행")
