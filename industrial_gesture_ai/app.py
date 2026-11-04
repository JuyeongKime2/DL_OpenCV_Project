"""
실시간 수신호 인식 Streamlit 앱.
학습된 모든 추출기(YOLOv8-Pose, Hands, Holistic)를 
한 화면에 나란히 배치하여 동시 비교합니다.
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
st.title("🏗️ AI 산업현장 작업자 수신호 인식 (다중 모델 비교)")
st.caption("YOLOv8-Pose / MediaPipe Hands / Holistic + LSTM 동시 추론")

# --- 사이드바 ---
st.sidebar.warning("시연용 프로토타입입니다. 실제 산업장비 제어에 사용하지 마세요.")
st.sidebar.info("💡 3개의 모델을 동시에 구동하므로 컴퓨터 사양에 따라 이전보다 반응 속도가 더 느려질 수 있습니다.")

# 모든 모델을 한 번에 메모리에 로드하는 함수로 변경
@st.cache_resource
def load_all_models():
    models = {}
    for ext_name in EXTRACTORS.keys():
        try:
            num_features = EXTRACTORS[ext_name]["num_features"]
            weight_path = MODEL_DIR / f"{ext_name}_lstm_best.pth"
            
            if not weight_path.exists():
                st.warning(f"⚠️ {ext_name} 모델 가중치가 없어 화면에서 제외됩니다.")
                continue
                
            model = GestureNet(num_features)
            model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            model.eval()
            extractor = get_extractor(ext_name)
            
            # 각 모델별로 독립적인 상태(buffer 등) 관리
            models[ext_name] = {
                "extractor": extractor,
                "model": model,
                "num_features": num_features,
                "buffer": deque(maxlen=SEQUENCE_LENGTH),
                "last_label": "대기 중",
                "last_conf": 0.0
            }
        except Exception as e:
            st.error(f"{ext_name} 로드 실패: {e}")
            
    return models

loaded_models = load_all_models()

if not loaded_models:
    st.error("불러올 수 있는 모델이 없습니다. 먼저 python train.py를 실행하세요.")
    st.stop()

run = st.checkbox("웹캠 시작", key="run")

# 모델 개수(보통 3개)만큼 화면을 세로 열(Column)로 균등 분할
cols = st.columns(len(loaded_models))
frame_areas = {}
result_areas = {}

# 각 컬럼에 제목과 UI 공간(empty) 할당
for i, (name, _) in enumerate(loaded_models.items()):
    with cols[i]:
        st.markdown(f"### {EXTRACTORS[name]['name']}")
        frame_areas[name] = st.empty()
        result_areas[name] = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        st.stop()

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("프레임을 읽을 수 없습니다.")
            break
            
        frame = cv2.flip(frame, 1)

        # 각 프레임을 로드된 모든 모델에 차례대로 통과시킴
        for name, p in loaded_models.items():
            
            # ✨ 중요: 뼈대 그리기(Annotation)가 겹치지 않도록 각 모델마다 깨끗한 프레임 복사본 제공
            frame_copy = frame.copy()
            
            extractor = p["extractor"]
            model = p["model"]
            num_features = p["num_features"]
            
            feature, annotated = extractor.extract(frame_copy)

            if len(feature) != num_features:
                feature = np.zeros(num_features, dtype=np.float32)
            p["buffer"].append(feature)

            if len(p["buffer"]) == SEQUENCE_LENGTH:
                seq = np.stack(p["buffer"]).astype(np.float32)
                with torch.no_grad():
                    logits = model(torch.tensor(seq).unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0].numpy()
                    pred = int(probs.argmax())
                    p["last_label"] = CLASS_KOREAN[pred]
                    p["last_conf"] = float(probs[pred])

            # 화면에 결과 표시 텍스트 그리기
            color = (0, 255, 0) if p["last_conf"] > 0.6 else (0, 165, 255)
            cv2.putText(annotated, f"{p['last_label']} ({p['last_conf']:.2f})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # 할당된 컬럼 위치에 이미지와 인식 결과 업데이트
            frame_areas[name].image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            result_areas[name].metric("인식 결과", p["last_label"], f"신뢰도 {p['last_conf']:.2f}")

    cap.release()

st.markdown("---")