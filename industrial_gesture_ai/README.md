# AI 기반 산업현장 작업자 수신호 인식 시스템
## 3가지 관절 추출 모델 비교 (YOLOv8-Pose / MediaPipe Hands / MediaPipe Holistic + LSTM)

## 1. 프로젝트 개요
산업현장(크레인, 건설, 제조)에서 사용되는 작업자 수신호(STOP / UP / DOWN)를 실시간으로 인식하는 AI 시스템입니다.

**핵심 비교 대상** (동일한 LSTM 분류기 사용):
| 추출기 | 특징 | Feature 차원 |
|--------|------|--------------|
| **YOLOv8-Pose** | 전신 17개 관절 | 51 |
| **MediaPipe Hands** | 손 21개 랜드마크 | 63 |
| **MediaPipe Holistic** | 전신 포즈 + 양손 | 225 |

> **💡 핵심 전처리: 좌표 정규화 (Normalization) 적용**
> 작업자의 카메라 거리 및 화면 위치 변화에 따른 과적합(Overfitting)을 방지하기 위해, 추출된 모든 랜드마크 좌표에 대해 중심점을 원점(0,0)으로 이동하고 스케일을 1로 통일하는 정규화 로직이 적용되어 있습니다 (`pose_utils.py`).

## 2. 프로젝트 구조 (Directory Structure)
```text
industrial_gesture_ai/
├── data/                     # 정규화 처리 후 추출기별로 수집된 시계열 데이터 저장소
├── models/                   # 학습 완료된 LSTM 모델 가중치(.pth) 보관
├── outputs/                  # 학습 결과(혼동행렬, csv) 및 성능 평가 시각화 산출물
├── app.py                    # Streamlit 기반 3개 모델 실시간 동시 추론 애플리케이션
├── collect_data.py           # 웹캠 동작 프레임 연속 추출 및 데이터 수집 스크립트
├── config.py                 # 전역 설정(클래스명, 시퀀스 길이, 디렉토리 경로 등) 관리
├── pose_utils.py             # 관절 추출기 인터페이스 및 랜드마크 좌표 정규화 로직
├── train.py                  # 데이터 로드, LSTM 학습 및 모델 성능 시각화 파이프라인
├── requirements.txt          # 프로젝트 실행을 위한 Python 패키지 의존성 목록
└── README.md                 # 프로젝트 개요 및 실행 가이드 문서
```

## 3. 설치
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 데이터 수집
각 추출기별로 별도 데이터를 수집합니다. (정규화가 적용되어 있으므로 기존 `data/` 및 `models/` 내역을 모두 삭제한 뒤 새로 수집해야 합니다.)
```bash
python collect_data.py --extractor yolo
python collect_data.py --extractor hands
python collect_data.py --extractor holistic
```
- `1` : STOP (손바닥을 카메라 쪽으로)
- `2` : UP (팔을 위로)
- `3` : DOWN (팔을 아래로)
- `q` : 종료

*권장: 클래스당 최소 100개 이상. 화면의 중앙/모서리, 카메라 앞/뒤 등 다양한 위치와 거리에서 다채롭게 수집하면 훨씬 견고해집니다.*

## 5. 모델 학습 및 비교
```bash
python train.py
```
세 가지 추출기 × LSTM을 학습하고 Macro F1 기준으로 성능을 비교합니다. 
결과는 `outputs/` 및 `models/` 폴더에 생성됩니다.
- 모델 가중치 `.pth` 파일
- 혼동행렬 PNG (Confusion Matrix)
- Classification Report JSON
- `model_comparison.csv` & 시각화 그래프

## 6. 실시간 인식 (다중 모델 동시 비교)
```bash
streamlit run app.py
```
- 학습된 3가지 모델(YOLO, Hands, Holistic)을 한 화면에 분할 렌더링하여 **실시간으로 동시에 추론하고 성능을 비교**할 수 있습니다.
- *(참고: 3개의 모델을 동시에 구동하므로 컴퓨터 사양에 따라 프레임(FPS) 저하가 발생할 수 있습니다.)*

## 7. 주의사항
- 시연 및 연구용 프로토타입입니다. 실제 산업장비 제어에 직접 연결하지 마세요.
- STOP 클래스는 안전과 직결되므로 혼동행렬에서 재현율(Recall)을 별도로 꼼꼼히 확인해야 합니다.
- 동일 세션에서 연속으로 잘라낸 지나치게 유사한 시퀀스 데이터가 Train/Val 셋에 섞여 들어가면 성능(Accuracy)이 과대평가될 수 있습니다.