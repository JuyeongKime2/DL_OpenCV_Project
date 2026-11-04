"""
세 가지 관절 추출기:
  - YOLOv8-Pose
  - MediaPipe Hands
  - MediaPipe Holistic

모두 extract(frame) → (feature_vector: np.ndarray, annotated_frame) 를 반환한다.
감지 실패 시 zero vector를 반환한다.
"""
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp

from config import YOLO_POSE_MODEL, EXTRACTORS


def _normalize_landmarks(landmarks, image_shape, has_z=True):
    """랜드마크를 0~1 상대좌표로 정규화. 반환 shape: (N, 3)"""
    h, w = image_shape[:2]
    points = []
    for lm in landmarks:
        x = lm.x
        y = lm.y
        z = getattr(lm, "z", 0.0) if has_z else 0.0
        # MediaPipe는 이미 0~1 정규화되어 있지만, 안전하게 clamp
        points.append([np.clip(x, 0, 1), np.clip(y, 0, 1), z])
    return np.array(points, dtype=np.float32)


class YoloPoseExtractor:
    def __init__(self, model_path=YOLO_POSE_MODEL):
        self.model = YOLO(model_path)
        self.num_features = EXTRACTORS["yolo"]["num_features"]

    def extract(self, frame):
        h, w = frame.shape[:2]
        result = self.model.predict(frame, verbose=False, conf=0.35)[0]
        annotated = result.plot()

        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return np.zeros(self.num_features, dtype=np.float32), annotated

        xy = result.keypoints.xy.cpu().numpy()
        # 가장 큰 사람 선택
        if result.boxes is not None and len(result.boxes) == len(xy):
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = int(np.argmax(areas))
        else:
            idx = 0

        points = xy[idx]
        conf = result.keypoints.conf.cpu().numpy()[idx] if result.keypoints.conf is not None else np.ones(17)

        vec = np.zeros((17, 3), dtype=np.float32)
        vec[:, 0] = points[:, 0] / max(w, 1)
        vec[:, 1] = points[:, 1] / max(h, 1)
        vec[:, 2] = conf
        return vec.reshape(-1), annotated


class MediaPipeHandsExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.num_features = EXTRACTORS["hands"]["num_features"]

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return np.zeros(self.num_features, dtype=np.float32), annotated

        hand_landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(
            annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
        )
        vec = _normalize_landmarks(hand_landmarks.landmark, frame.shape).reshape(-1)
        return vec, annotated

    def __del__(self):
        if hasattr(self, "hands"):
            self.hands.close()


class MediaPipeHolisticExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.num_features = EXTRACTORS["holistic"]["num_features"]

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        annotated = frame.copy()

        features = []

        # Pose (33)
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
            features.append(_normalize_landmarks(results.pose_landmarks.landmark, frame.shape))
        else:
            features.append(np.zeros((33, 3), dtype=np.float32))

        # Left hand
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(
                annotated, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            features.append(_normalize_landmarks(results.left_hand_landmarks.landmark, frame.shape))
        else:
            features.append(np.zeros((21, 3), dtype=np.float32))

        # Right hand
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                annotated, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            features.append(_normalize_landmarks(results.right_hand_landmarks.landmark, frame.shape))
        else:
            features.append(np.zeros((21, 3), dtype=np.float32))

        vec = np.concatenate(features, axis=0).reshape(-1)
        return vec, annotated

    def __del__(self):
        if hasattr(self, "holistic"):
            self.holistic.close()


def get_extractor(name: str):
    name = name.lower()
    if name == "yolo":
        return YoloPoseExtractor()
    elif name == "hands":
        return MediaPipeHandsExtractor()
    elif name == "holistic":
        return MediaPipeHolisticExtractor()
    else:
        raise ValueError(f"Unknown extractor: {name}. Choose from {list(EXTRACTORS.keys())}")


# 하위 호환성을 위한 별칭
PoseExtractor = YoloPoseExtractor
