"""
데이터 수집 스크립트.
사용법:
  python collect_data.py --extractor yolo
  python collect_data.py --extractor hands
  python collect_data.py --extractor holistic
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

from pose_utils import get_extractor
from config import DATA_DIR, SEQUENCE_LENGTH, CLASS_NAMES, EXTRACTORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", type=str, default="yolo",
                        choices=list(EXTRACTORS.keys()),
                        help="관절 추출 방식 (yolo / hands / holistic)")
    args = parser.parse_args()

    extractor_name = args.extractor
    extractor = get_extractor(extractor_name)
    num_features = EXTRACTORS[extractor_name]["num_features"]

    save_root = DATA_DIR / extractor_name
    save_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    sequence = []
    saved = {i: 0 for i in CLASS_NAMES}

    print(f"=== {EXTRACTORS[extractor_name]['name']} 데이터 수집 ===")
    print("1=STOP, 2=UP, 3=DOWN 키를 누르는 순간 최근 30프레임을 저장합니다.")
    print("q=종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feature, annotated = extractor.extract(frame)

        if len(feature) != num_features:
            feature = np.zeros(num_features, dtype=np.float32)

        sequence.append(feature)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        key = cv2.waitKey(1) & 0xFF
        label = None
        if key == ord("1"):
            label = 0
        elif key == ord("2"):
            label = 1
        elif key == ord("3"):
            label = 2

        if label is not None and len(sequence) == SEQUENCE_LENGTH:
            x = np.asarray(sequence, dtype=np.float32)
            class_dir = save_root / CLASS_NAMES[label]
            class_dir.mkdir(parents=True, exist_ok=True)
            saved[label] += 1
            np.save(class_dir / f"sample_{saved[label]:05d}.npy", x)
            print(f"저장: {CLASS_NAMES[label]} / 총 {saved[label]}개")

        cv2.putText(annotated, f"[{extractor_name.upper()}] 1:STOP 2:UP 3:DOWN | q:quit",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, name in CLASS_NAMES.items():
            cv2.putText(annotated, f"{name}: {saved[i]}",
                        (15, 90 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("Gesture Data Collection", annotated)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("수집 종료. 저장 개수:", saved)


if __name__ == "__main__":
    main()
