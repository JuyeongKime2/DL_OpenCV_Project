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

## 2. 설치
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. 데이터 수집
각 추출기별로 별도 데이터를 수집합니다.
```bash
python collect_data.py --extractor yolo
python collect_data.py --extractor hands
python collect_data.py --extractor holistic
```
- `1` : STOP (손바닥을 카메라 쪽으로)
- `2` : UP (팔을 위로)
- `3` : DOWN (팔을 아래로)
- `q` : 종료

권장: 클래스당 100개 이상. 다양한 거리·조명·속도로 수집하세요.

## 4. 모델 학습 및 비교
```bash
python train.py
```
세 가지 추출기 × LSTM을 학습하고 Macro F1 기준으로 비교합니다.
결과는 `outputs/`에 저장됩니다.
- 혼동행렬 PNG
- classification report JSON
- `model_comparison.csv`

## 5. 실시간 인식
```bash
streamlit run app.py
```
사이드바에서 추출기/모델을 선택할 수 있습니다.

## 6. 주의사항
- 시연 및 연구용입니다. 실제 산업장비 제어에 연결하지 마세요.
- STOP 클래스는 안전과 관련되므로 Recall을 별도로 확인하세요.
- 동일 세션에서 잘라낸 유사 시퀀스가 train/val에 동시에 들어가면 성능이 과대평가될 수 있습니다.
