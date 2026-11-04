
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# 관절 특징 추출기별 입력 크기
EXTRACTORS = {
    "yolo": {
        "name": "YOLOv8-Pose",
        "num_features": 17 * 3,
        "description": "전신 17개 관절 특징",
    },
    "hands": {
        "name": "MediaPipe Hands",
        "num_features": 21 * 3,
        "description": "손 21개 랜드마크 특징",
    },
    "holistic": {
        "name": "MediaPipe Holistic",
        "num_features": (33 + 21 + 21) * 3,
        "description": "전신 포즈 + 양손 랜드마크 특징",
    },
}

DEFAULT_EXTRACTOR = "yolo"
SEQUENCE_LENGTH = 30
YOLO_POSE_MODEL = "yolov8n-pose.pt"

# 제스처 클래스
CLASS_NAMES = {
    0: "STOP",
    1: "UP",
    2: "DOWN",
    3: "OTHER",
}

CLASS_KOREAN = {
    0: "작업 정지",
    1: "장비 상승",
    2: "장비 하강",
    3: "수신호 아님 / 기타",
}

NUM_CLASSES = len(CLASS_NAMES)

# 수집 단계의 최소 관절 검출 비율
MIN_VALID_FRAME_RATIO = 0.80

# 실시간 인식 안정화
MIN_MODEL_SCORE = 0.60
SMOOTHING_WINDOW = 5
MIN_CONSISTENT_VOTES = 3
PREDICTION_INTERVAL = 5

# 세션 독립 평가 설정
MIN_SESSION_COUNT = 1
MIN_TRAIN_SAMPLES_PER_CLASS = 5
MIN_EVAL_SAMPLES_PER_CLASS = 2
