"""
산업현장 작업자 수신호 데이터 수집기

실행:
    # 1. MediaPipe Pose (전신) 데이터 수집
    python collect_data.py --extractor pose

    # 2. MediaPipe Dual Hands (양손) 데이터 수집
    python collect_data.py --extractor dual_hands

    # 3. MediaPipe Holistic (전신+양손) 데이터 수집
    python collect_data.py --extractor holistic

조작:
    1 : STOP
    2 : UP
    3 : EMERGENCY
    q : 종료

주요 기능:
    - 30프레임 시퀀스 수집
    - 저장 후 버퍼 초기화
    - 관절 검출률 검사
    - 세션 단위 데이터 저장
    - Pillow를 이용한 한글 출력
"""

import argparse
import json
import time

from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from config import (
    CLASS_NAMES,
    DATA_DIR,
    EXTRACTORS,
    MIN_VALID_FRAME_RATIO,
    SEQUENCE_LENGTH,
)

from pose_utils import (
    get_extractor,
    has_valid_detection,
)


# ============================================================
# 1. 키보드 및 클래스 설정
# ============================================================

KEY_TO_LABEL = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
}


# ============================================================
# 2. 한글 폰트 불러오기
# ============================================================

def load_korean_font(font_size=20, custom_path=None):

    """
    Windows 한글 폰트를 불러온다.
    """

    candidates = []

    if custom_path:
        candidates.append(custom_path)

    candidates.extend([
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NanumGothicBold.ttf",
    ])

    for font_path in candidates:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(
                    font_path,
                    font_size
                )
            except OSError:
                continue

    raise FileNotFoundError(
        "한글 폰트를 찾을 수 없습니다.\n"
        "Windows의 맑은 고딕 폰트 설치 여부를 확인하세요."
    )


# ============================================================
# 3. OpenCV 프레임에 한글 출력
# ============================================================

def draw_korean_text(
    frame,
    text,
    position=(15, 200),
    font_size=20,
    color=(255, 255, 255),
    font=None,
):
    """
    OpenCV 영상에 Pillow를 이용해 한글을 출력한다.

    OpenCV 기본 putText()는 한글을 지원하지 않기 때문에
    PIL ImageDraw를 사용한다.
    """

    if font is None:
        font = load_korean_font(font_size)

    # OpenCV BGR -> PIL RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    x, y = position

    # 화면 너비에 맞춰 자동 줄바꿈
    max_width = image.width - x - 20
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        candidate = (f"{current_line} {word}".strip())
        bbox = draw.textbbox((0, 0), candidate, font=font)
        text_width = (bbox[2] - bbox[0])

        if (text_width > max_width and current_line):
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate

    if current_line:
        lines.append(current_line)

    final_text = "\n".join(lines)

    # 가독성을 위한 어두운 배경 (패딩 축소 적용)
    bbox = draw.multiline_textbbox(
        (x, y),
        final_text,
        font=font,
        spacing=5
    )

    padding = 2  # 기존 5에서 2로 축소하여 박스 크기를 줄임

    draw.rounded_rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ),
        radius=3, # 모서리 둥글기도 비례하여 축소
        fill=(25, 25, 25)
    )

    draw.multiline_text(
        (x, y),
        final_text,
        font=font,
        fill=color,
        spacing=5
    )

    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# 4. 카메라 초기화
# ============================================================

def open_camera(camera_index=0):

    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap.release()
        # 기본 방식으로 재시도
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "웹캠을 열 수 없습니다. "
            "카메라 연결 및 권한을 확인하세요."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap


# ============================================================
# 5. 데이터 수집
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="산업현장 수신호 AI 데이터 수집"
    )

    parser.add_argument(
        "--extractor",
        type=str,
        default="pose",
        choices=list(EXTRACTORS.keys()),
        help="관절 추출 방식"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="웹캠 장치 번호"
    )

    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="한글 폰트 파일 경로"
    )

    args = parser.parse_args()

    extractor_name = args.extractor

    num_features = EXTRACTORS[extractor_name]["num_features"]

    # 한글 폰트 미리 로드
    korean_font = load_korean_font(
        font_size=20,
        custom_path=args.font
    )

    small_font = load_korean_font(
        font_size=16,
        custom_path=args.font
    )

    # 관절 추출기 초기화
    extractor = get_extractor(extractor_name)

    # ========================================================
    # 6. 독립 촬영 세션 생성
    # ========================================================

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    session_dir = (
        DATA_DIR
        / extractor_name
        / "sessions"
        / session_id
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 7. 카메라 실행
    # ========================================================

    try:
        cap = open_camera(args.camera)
    except Exception:
        if hasattr(extractor, "close"):
            extractor.close()
        raise

    # 최근 30프레임 저장 버퍼
    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 각 프레임의 관절 검출 여부
    valid_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # 클래스별 저장 개수
    saved_counts = {
        name: 0
        for name in CLASS_NAMES.values()
    }

    # 화면 상태 메시지
    last_message = "웹캠 앞에서 수신호를 보여주세요."

    detection_status = False
    detection_ratio = 0.0
    frame_count = 0

    # ========================================================
    # 8. 수집 안내 출력
    # ========================================================

    print("=" * 60)
    print(f"관절 추출 모델: {EXTRACTORS[extractor_name]['name']}")
    print(f"세션 ID: {session_id}")
    print("=" * 60)
    print("1 = STOP")
    print("2 = UP")
    print("3 = EMERGENCY")
    print("q = QUIT")
    print("-" * 60)
    print("30프레임이 모이면 숫자 키를 눌러 저장하세요.")
    print("저장 후에는 버퍼가 초기화됩니다.")

    if extractor_name == "dual_hands":
        print("MediaPipe Dual Hands는 카메라 안에 보이는 손을 검출해야 합니다.")

    print("=" * 60)

    # ========================================================
    # 9. 실시간 데이터 수집 루프
    # ========================================================

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                last_message = "웹캠 프레임을 읽지 못했습니다."
                break

            # 좌우 반전
            frame = cv2.flip(frame, 1)

            # 관절 특징 추출
            feature, annotated = extractor.extract(frame)
            feature = np.asarray(feature, dtype=np.float32)

            # =================================================
            # 관절 검출 여부 검사
            # =================================================

            if (feature.size != num_features or not np.isfinite(feature).all()):
                feature = np.zeros(num_features, dtype=np.float32)
                is_valid = False
            else:
                is_valid = has_valid_detection(feature, extractor_name)

            detection_status = is_valid

            # =================================================
            # 프레임 버퍼 저장
            # =================================================

            feature_buffer.append(feature)
            valid_buffer.append(is_valid)
            frame_count += 1

            # =================================================
            # 키보드 입력
            # =================================================

            key = cv2.waitKey(1) & 0xFF
            label = KEY_TO_LABEL.get(key)

            # =================================================
            # 데이터 저장
            # =================================================

            if label is not None:
                # 30프레임 확보 여부
                if len(feature_buffer) < SEQUENCE_LENGTH:
                    last_message = (
                        f"프레임 부족: "
                        f"{len(feature_buffer)}/"
                        f"{SEQUENCE_LENGTH}"
                    )
                else:
                    sequence = np.stack(feature_buffer).astype(np.float32)
                    detection_ratio = float(np.mean(valid_buffer))
                    class_name = CLASS_NAMES[label]

                    if not np.isfinite(sequence).all():
                        last_message = "잘못된 특징값이 포함되어 저장하지 않았습니다."

                    elif (detection_ratio < MIN_VALID_FRAME_RATIO):
                        if extractor_name == "dual_hands":
                            last_message = (
                                "손이 충분히 검출되지 않았습니다. "
                                "손 전체를 카메라 안에 보여주세요."
                            )
                        else:
                            last_message = (
                                "관절 검출률이 부족합니다. "
                                "촬영 위치를 조정해주세요."
                            )
                    else:
                        # 클래스별 폴더 생성
                        class_dir = session_dir / class_name
                        class_dir.mkdir(parents=True, exist_ok=True)

                        # 파일 번호 증가
                        next_number = saved_counts[class_name] + 1
                        file_path = class_dir / f"sample_{next_number:05d}.npy"

                        # 데이터 저장
                        np.save(file_path, sequence)
                        saved_counts[class_name] = next_number

                        last_message = (
                            f"{class_name} 데이터 저장 완료 "
                            f"({next_number}개)"
                        )
                        print(last_message)

                    # =================================================
                    # 중요: 저장 시 버퍼 초기화
                    # =================================================
                    feature_buffer.clear()
                    valid_buffer.clear()

            # =================================================
            # 10. 화면 시각화
            # =================================================

            # 제목 (OTHER 삭제 및 변경된 클래스 반영)
            cv2.putText(
                annotated,
                (
                    f"[{extractor_name.upper()}] "
                    "1:STOP  2:UP  3:EMERGENCY  q:QUIT"
                ),
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            # 버퍼 상태
            cv2.putText(
                annotated,
                f"Buffer: {len(feature_buffer)}/{SEQUENCE_LENGTH}",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # 관절 검출 상태
            detection_text = (
                "Detection: OK"
                if detection_status
                else "Detection: NOT DETECTED"
            )

            detection_color = (
                (0, 255, 0)
                if detection_status
                else (0, 0, 255)
            )

            cv2.putText(
                annotated,
                detection_text,
                (15, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                detection_color,
                2
            )

            # 검출 비율
            if len(valid_buffer) > 0:
                detection_ratio = float(np.mean(valid_buffer))

            cv2.putText(
                annotated,
                f"Detection Rate: {detection_ratio:.0%}",
                (15, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                1
            )

            # 클래스별 저장 개수
            for i, class_name in enumerate(CLASS_NAMES.values()):
                cv2.putText(
                    annotated,
                    f"{class_name}: {saved_counts[class_name]}",
                    (15, 195 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1
                )

            # =================================================
            # 11. 한글 메시지 출력
            # =================================================

            if not detection_status and extractor_name == "dual_hands":
                status_message = "손 검출 대기 중: 카메라 안에 손 전체를 보여주세요."
                message_color = (255, 100, 100)
            else:
                status_message = last_message
                message_color = (255, 255, 255)

            annotated = draw_korean_text(
                annotated,
                status_message,
                position=(15, 350),
                font_size=20,
                color=message_color,
                font=korean_font
            )

            # 화면 표시
            cv2.imshow("Gesture Data Collection", annotated)

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hasattr(extractor, "close"):
            extractor.close()

    # ========================================================
    # 12. 수집 결과 저장
    # ========================================================

    session_info = {
        "session_id": session_id,
        "extractor": extractor_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": num_features,
        "saved_samples": saved_counts,
        "total_samples": int(sum(saved_counts.values())),
        "minimum_valid_frame_ratio": MIN_VALID_FRAME_RATIO,
        "note": "독립 촬영 세션. 동일 세션의 샘플은 하나의 데이터 분할에만 사용."
    }

    info_path = session_dir / "session_info.json"
    info_path.write_text(
        json.dumps(session_info, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("데이터 수집 종료")
    print("=" * 60)

    for class_name, count in saved_counts.items():
        print(f"{class_name}: {count}개")

    print(f"총 수집 개수: {sum(saved_counts.values())}개")
    print(f"저장 경로: {session_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()