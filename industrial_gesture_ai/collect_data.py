
"""독립 촬영 세션 기반 산업현장 수신호 데이터 수집기."""

import argparse
import json

from collections import deque
from datetime import datetime

import cv2
import numpy as np

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


KEY_TO_LABEL = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
    ord("4"): 3,
}


def main():

    parser = argparse.ArgumentParser(
        description="독립 세션 기반 수신호 데이터 수집"
    )

    parser.add_argument(
        "--extractor",
        type=str,
        default="yolo",
        choices=list(EXTRACTORS.keys()),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="웹캠 장치 번호",
    )

    args = parser.parse_args()

    extractor_name = args.extractor

    extractor = get_extractor(extractor_name)

    num_features = EXTRACTORS[extractor_name]["num_features"]

    # 실행할 때마다 새로운 세션 생성
    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

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

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():

        if hasattr(extractor, "close"):
            extractor.close()

        raise RuntimeError(
            f"웹캠을 열 수 없습니다. 카메라 번호 확인: {args.camera}"
        )

    buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    valid_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    saved_counts = {
        name: 0
        for name in CLASS_NAMES.values()
    }

    last_message = (
        "동작을 취하고 30프레임이 모이면 숫자 키를 누르세요."
    )

    print("=" * 60)
    print(
        f"데이터 수집 방식: {EXTRACTORS[extractor_name]['name']}"
    )
    print(f"세션 ID: {session_id}")

    print("1 = STOP")
    print("2 = UP")
    print("3 = DOWN")
    print("4 = OTHER")
    print("q = 종료")

    print("저장할 때마다 버퍼가 초기화됩니다.")
    print("다른 촬영 환경은 종료 후 다시 실행해서 수집하세요.")
    print("=" * 60)

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                last_message = "웹캠 프레임을 읽지 못했습니다."
                break

            frame = cv2.flip(frame, 1)

            feature, annotated = extractor.extract(frame)

            if (
                len(feature) != num_features
                or not np.isfinite(feature).all()
            ):

                feature = np.zeros(
                    num_features,
                    dtype=np.float32
                )

                is_valid = False

            else:

                feature = np.asarray(
                    feature,
                    dtype=np.float32
                )

                is_valid = has_valid_detection(
                    feature,
                    extractor_name
                )

            buffer.append(feature)
            valid_buffer.append(is_valid)

            key = cv2.waitKey(1) & 0xFF

            label = KEY_TO_LABEL.get(key)

            if label is not None:

                if len(buffer) < SEQUENCE_LENGTH:

                    last_message = (
                        f"프레임 부족: "
                        f"{len(buffer)}/{SEQUENCE_LENGTH}"
                    )

                else:

                    sequence = np.stack(
                        buffer
                    ).astype(np.float32)

                    valid_ratio = float(
                        np.mean(valid_buffer)
                    )

                    class_name = CLASS_NAMES[label]

                    if not np.isfinite(sequence).all():

                        last_message = (
                            "NaN/Inf 특징값이 포함되어 저장하지 않았습니다."
                        )

                    elif valid_ratio < MIN_VALID_FRAME_RATIO:

                        last_message = (
                            f"관절 검출 프레임 부족: "
                            f"{valid_ratio:.0%}. "
                            "카메라에 동작이 잘 보이도록 조정하세요."
                        )

                    else:

                        class_dir = (
                            session_dir / class_name
                        )

                        class_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        next_number = (
                            saved_counts[class_name] + 1
                        )

                        file_path = (
                            class_dir
                            / f"sample_{next_number:05d}.npy"
                        )

                        np.save(
                            file_path,
                            sequence
                        )

                        saved_counts[class_name] = next_number

                        last_message = (
                            f"저장 완료: {class_name} "
                            f"/ {next_number}개 "
                            f"(검출률 {valid_ratio:.0%})"
                        )

                        print(last_message)

                    # 핵심: 저장 성공/실패와 관계없이 버퍼 초기화
                    buffer.clear()
                    valid_buffer.clear()

            # 화면 출력
            cv2.putText(
                annotated,
                "1:STOP  2:UP  3:DOWN  4:OTHER  q:QUIT",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Detection: {'OK' if is_valid else 'NOT DETECTED'}",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0) if is_valid else (0, 0, 255),
                2,
            )

            y = 120

            for class_name, count in saved_counts.items():

                cv2.putText(
                    annotated,
                    f"{class_name}: {count}",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 0),
                    1,
                )

                y += 22

            cv2.putText(
                annotated,
                last_message[:100],
                (12, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "Gesture Data Collection",
                annotated
            )

            if key == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        if hasattr(extractor, "close"):
            extractor.close()

    # 세션 메타데이터 저장
    session_info = {

        "session_id": session_id,

        "extractor": extractor_name,

        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "sequence_length": SEQUENCE_LENGTH,

        "saved_samples": saved_counts,

        "note": (
            "한 번의 collect_data.py 실행을 "
            "하나의 독립 세션으로 취급합니다."
        ),
    }

    (session_dir / "session_info.json").write_text(

        json.dumps(
            session_info,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )

    print("\n수집 종료")
    print("저장 개수:", saved_counts)
    print("세션 경로:", session_dir)

    print(
        "각 세션에서 4개 클래스를 모두 수집하고, "
        "최소 5개 세션을 확보하세요."
    )


if __name__ == "__main__":
    main()
