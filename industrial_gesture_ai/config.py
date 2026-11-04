from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# 세 가지 관절 추출 방식
EXTRACTORS = {
    "yolo": {
        "name": "YOLOv8-Pose",
        "num_features": 17 * 3,          # COCO 17 keypoints × (x, y, conf)
        "description": "전신 17개 관절 (어깨·팔꿈치·손목 포함)"
    },
    "hands": {
        "name": "MediaPipe Hands",
        "num_features": 21 * 3,          # 21 hand landmarks × (x, y, z)
        "description": "손 21개 랜드마크 (손가락 모양까지 정밀)"
    },
    "holistic": {
        "name": "MediaPipe Holistic",
        "num_features": (33 + 21 + 21) * 3,  # pose 33 + left hand 21 + right hand 21 × (x,y,z)
        "description": "전신 포즈 + 양손 랜드마크"
    },
}

DEFAULT_EXTRACTOR = "yolo"
SEQUENCE_LENGTH = 30
STRIDE = 15
YOLO_POSE_MODEL = "yolov8n-pose.pt"

# 클래스 정의
CLASS_NAMES = {
    0: "STOP",
    1: "UP",
    2: "DOWN",
}
CLASS_KOREAN = {
    0: "작업 정지",
    1: "장비 상승",
    2: "장비 하강",
}

NUM_CLASSES = 3
