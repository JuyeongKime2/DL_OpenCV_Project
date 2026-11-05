"""YOLOv8-Pose / MediaPipe Hands / MediaPipe Holistic 특징 추출 유틸리티."""

import cv2
import mediapipe as mp
import numpy as np

from ultralytics import YOLO

from config import EXTRACTORS, YOLO_POSE_MODEL


def normalize_features(feature_array, stride=3, valid_mask=None):
    """좌표의 중심과 크기를 정규화한다.
    누락된 관절은 계산에서 제외한다.
    세 번째 채널(confidence 또는 z)은 그대로 유지한다.
    """

    arr = np.asarray(feature_array, dtype=np.float32)

    if arr.size == 0 or arr.size % stride != 0:
        return arr

    points = arr.reshape(-1, stride).copy()
    xy = points[:, :2]

    valid = np.isfinite(xy).all(axis=1)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)

        if len(mask) != len(points):
            raise ValueError("valid_mask 길이가 관절 개수와 일치하지 않습니다.")

        valid &= mask

    else:
        valid &= np.any(np.abs(xy) > 1e-8, axis=1)

    if valid.sum() < 2:
        points[~valid, :2] = 0.0
        return points.reshape(-1)

    center = np.median(xy[valid], axis=0)
    centered = xy[valid] - center

    scale = float(
        np.max(np.linalg.norm(centered, axis=1))
    )

    if scale > 1e-8:
        points[valid, :2] = centered / scale
    else:
        points[valid, :2] = 0.0

    points[~valid, :2] = 0.0

    return points.reshape(-1)


def _normalize_landmarks(landmarks):
    """MediaPipe 랜드마크를 (N, 3) 배열로 변환한다."""

    points = []

    for lm in landmarks:
        points.append([
            float(np.clip(lm.x, 0.0, 1.0)),
            float(np.clip(lm.y, 0.0, 1.0)),
            float(getattr(lm, "z", 0.0)),
        ])

    return np.asarray(points, dtype=np.float32)


# ============================================================
# MediaPipe Pose (새로 추가)
# ============================================================
class MediaPipePoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.num_features = EXTRACTORS["pose"]["num_features"]

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        annotated = frame.copy()

        if not results.pose_landmarks:
            return np.zeros(self.num_features, dtype=np.float32), annotated

        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )
        points = _normalize_landmarks(results.pose_landmarks.landmark)
        features = normalize_features(points.reshape(-1), stride=3)

        return features.astype(np.float32), annotated

    def close(self):
        if getattr(self, "pose", None) is not None:
            self.pose.close()
            self.pose = None

# ============================================================
# MediaPipe Dual Hands (기존 Hands 대체 - 좌우 손 고정 적용)
# ============================================================
class MediaPipeDualHandsExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.num_features = EXTRACTORS["dual_hands"]["num_features"]

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()

        # 양손 데이터를 담을 0 배열 준비 [0슬롯: 왼손, 1슬롯: 오른손] (2손 x 21관절 x 3좌표)
        features = np.zeros((2, 21, 3), dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if idx >= 2:
                    break
                
                # MediaPipe의 classification label 확인 ("Left" 또는 "Right")
                # 주의: 화면 기준이므로 카메라 반전 여부에 따라 레이블이 다를 수 있으나, 
                # 일반적으로 multi_handedness는 카메라 센서 기준 좌/우를 반환합니다.
                handedness_label = results.multi_handedness[idx].classification[0].label
                
                # 좌우 손 구분에 따른 슬롯 고정 (Left -> 0번 슬롯, Right -> 1번 슬롯)
                # 만약 화면 좌우 반전으로 반대로 잡힌다면 아래 조건을 조정하세요.
                slot_idx = 0 if handedness_label == "Left" else 1

                mp.solutions.drawing_utils.draw_landmarks(
                    annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                points = _normalize_landmarks(hand_landmarks.landmark)
                features[slot_idx] = normalize_features(points.reshape(-1), stride=3).reshape(21, 3)

        return features.reshape(-1), annotated

    def close(self):
        if getattr(self, "hands", None) is not None:
            self.hands.close()
            self.hands = None


# ============================================================
# MediaPipe Holistic
# ============================================================

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

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.holistic.process(rgb)

        annotated = frame.copy()

        groups = []

        # 전신 Pose
        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )

            pose = _normalize_landmarks(
                results.pose_landmarks.landmark
            )

            groups.append(
                normalize_features(
                    pose.reshape(-1),
                    stride=3
                ).reshape(33, 3)
            )

        else:
            groups.append(
                np.zeros((33, 3), dtype=np.float32)
            )

        # 왼손
        if results.left_hand_landmarks:

            self.mp_draw.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            left = _normalize_landmarks(
                results.left_hand_landmarks.landmark
            )

            groups.append(
                normalize_features(
                    left.reshape(-1),
                    stride=3
                ).reshape(21, 3)
            )

        else:
            groups.append(
                np.zeros((21, 3), dtype=np.float32)
            )

        # 오른손
        if results.right_hand_landmarks:

            self.mp_draw.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )

            right = _normalize_landmarks(
                results.right_hand_landmarks.landmark
            )

            groups.append(
                normalize_features(
                    right.reshape(-1),
                    stride=3
                ).reshape(21, 3)
            )

        else:
            groups.append(
                np.zeros((21, 3), dtype=np.float32)
            )

        return (
            np.concatenate(groups, axis=0).reshape(-1),
            annotated
        )

    def close(self):

        if getattr(self, "holistic", None) is not None:

            self.holistic.close()
            self.holistic = None

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass


# ============================================================
# 관절 검출 유효성 확인 (기준 강화)
# ============================================================

def has_valid_detection(features, extractor_name):
    arr = np.asarray(features, dtype=np.float32)
    if arr.size != EXTRACTORS[extractor_name]["num_features"]:
        return False
    if not np.isfinite(arr).all():
        return False

    if extractor_name == "pose":
        # Pose는 상체/전신 관절 중 최소 10개 이상이 유효하게 잡혀야 함
        points = arr.reshape(33, 3)
        return int(np.sum(np.linalg.norm(points[:, :2], axis=1) > 1e-6)) >= 10

    if extractor_name == "dual_hands":
        # Dual Hands는 양손 합쳐서 최소 10개 이상의 손가락 관절이 잡혀야 함 (노이즈 방지)
        points = arr.reshape(42, 3)
        return int(np.sum(np.linalg.norm(points[:, :2], axis=1) > 1e-6)) >= 10

    if extractor_name == "holistic":
        # Holistic은 전신(33개) 외에 '손 관절(좌우 각 21개)' 중 적어도 한쪽 손이라도
        # 유의미하게(예: 손가락 관절 5개 이상) 잡혀야 유효 프레임으로 인정
        points = arr.reshape(75, 3)
        
        # 슬롯별 분리: Pose(0~32), Left Hand(33~53), Right Hand(54~74)
        left_hand_points = points[33:54, :2]
        right_hand_points = points[54:75, :2]
        
        left_active = int(np.sum(np.linalg.norm(left_hand_points, axis=1) > 1e-6))
        right_active = int(np.sum(np.linalg.norm(right_hand_points, axis=1) > 1e-6))
        
        # 적어도 한쪽 손에서 5개 이상의 관절이 검출되어야 함 (양손 수신호 품질 보장)
        return (left_active >= 5) or (right_active >= 5)

    return False


# ============================================================
# 추출기 호출
# ============================================================

def get_extractor(name: str):
    name = name.lower()
    if name == "pose": return MediaPipePoseExtractor()
    if name == "dual_hands": return MediaPipeDualHandsExtractor()
    if name == "holistic": return MediaPipeHolisticExtractor()
    raise ValueError(f"알 수 없는 추출기: {name}")