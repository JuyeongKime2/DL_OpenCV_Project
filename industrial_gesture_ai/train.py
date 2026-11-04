
"""세션 독립 분할 기반 LSTM 학습 및 최종 테스트 평가."""

import argparse
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_NAMES,
    DATA_DIR,
    EXTRACTORS,
    MIN_EVAL_SAMPLES_PER_CLASS,
    MIN_SESSION_COUNT,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    SEQUENCE_LENGTH,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 96
PATIENCE = 10
RANDOM_SEED = 42


# ============================================================
# 1. 재현성 설정
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 데이터셋
# ============================================================

class GestureDataset(Dataset):

    def __init__(self, x, y):

        self.x = torch.tensor(
            x,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return self.x[index], self.y[index]


# ============================================================
# 3. LSTM 모델
# ============================================================

class GestureNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden=HIDDEN_SIZE,
        classes=NUM_CLASSES
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden,

            num_layers=2,

            batch_first=True,

            dropout=0.2,
        )

        self.classifier = nn.Sequential(

            nn.Linear(hidden, 64),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(64, classes),
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        return self.classifier(
            output[:, -1, :]
        )


# ============================================================
# 4. 세션 단위 데이터 로드
# ============================================================

def load_data(extractor_name):

    num_features = EXTRACTORS[
        extractor_name
    ]["num_features"]

    data_root = DATA_DIR / extractor_name

    sessions_root = (
        data_root / "sessions"
    )

    legacy_files = []

    if data_root.exists():

        for class_name in CLASS_NAMES.values():

            legacy_files.extend(
                (data_root / class_name).glob("*.npy")
            )

    if legacy_files:

        print(
            f"[안내] 기존 flat 형식 데이터 "
            f"{len(legacy_files)}개를 발견했습니다. "
            "세션 분할이 불가능하므로 학습에서 제외합니다."
        )

    if not sessions_root.exists():

        raise RuntimeError(

            f"세션 데이터 폴더가 없습니다: {sessions_root}\n"

            f"먼저 python collect_data.py "
            f"--extractor {extractor_name} 를 실행하세요."
        )

    x_list = []
    y_list = []
    group_list = []
    path_list = []

    session_dirs = sorted(

        path
        for path in sessions_root.iterdir()
        if path.is_dir()
    )

    for session_dir in session_dirs:

        session_id = session_dir.name

        for label_id, class_name in CLASS_NAMES.items():

            class_dir = session_dir / class_name

            if not class_dir.exists():
                continue

            for file_path in sorted(
                class_dir.glob("*.npy")
            ):

                try:

                    sample = np.load(
                        file_path,
                        allow_pickle=False
                    )

                except Exception as exc:

                    print(
                        f"[제외] 파일을 읽지 못함: "
                        f"{file_path} ({exc})"
                    )

                    continue

                if sample.shape != (
                    SEQUENCE_LENGTH,
                    num_features
                ):

                    print(
                        f"[제외] 데이터 크기 오류: "
                        f"{file_path.name}, "
                        f"현재={sample.shape}, "
                        f"기대={(SEQUENCE_LENGTH, num_features)}"
                    )

                    continue

                if not np.isfinite(sample).all():

                    print(
                        f"[제외] NaN/Inf 포함: {file_path}"
                    )

                    continue

                x_list.append(
                    sample.astype(np.float32)
                )

                y_list.append(label_id)

                group_list.append(session_id)

                path_list.append(str(file_path))

    if not x_list:

        raise RuntimeError(
            "학습 가능한 세션 데이터가 없습니다. "
            "데이터 수집을 먼저 진행하세요."
        )

    x = np.asarray(
        x_list,
        dtype=np.float32
    )

    y = np.asarray(
        y_list,
        dtype=np.int64
    )

    groups = np.asarray(group_list)

    return (
        x,
        y,
        groups,
        num_features,
        path_list
    )


# ============================================================
# 5. 세션 분할 보조 함수
# ============================================================

def _class_counts(y, indices):

    return np.bincount(
        y[indices],
        minlength=NUM_CLASSES
    )


def _has_minimum_samples(
    y,
    indices,
    minimum
):

    return bool(
        np.all(
            _class_counts(y, indices) >= minimum
        )
    )


# ============================================================
# 6. 세션 단위 Train / Validation / Test 분할
# ============================================================

def split_by_session(
    x,
    y,
    groups,
    seed=RANDOM_SEED
):

    unique_sessions = np.unique(groups)

    session_count = len(unique_sessions)

    if session_count < MIN_SESSION_COUNT:

        raise RuntimeError(

            f"독립 수집 세션이 {session_count}개입니다. "
            f"최소 {MIN_SESSION_COUNT}개가 필요합니다.\n"

            "collect_data.py를 여러 번 실행하고, "
            "매번 위치/거리/각도 등 촬영 조건을 바꿔 주세요."
        )

    # 전체 세션의 약 20%를 Test, 약 20%를 Validation에 할당
    test_session_count = max(
        1,
        int(round(session_count * 0.20))
    )

    val_session_count = max(
        1,
        int(round(session_count * 0.20))
    )

    if test_session_count + val_session_count >= session_count:

        test_session_count = 1
        val_session_count = 1

    for attempt in range(2000):

        rng = np.random.default_rng(
            seed + attempt
        )

        shuffled = rng.permutation(
            unique_sessions
        )

        test_sessions = shuffled[
            :test_session_count
        ]

        val_sessions = shuffled[
            test_session_count:
            test_session_count + val_session_count
        ]

        train_sessions = shuffled[
            test_session_count + val_session_count:
        ]

        train_idx = np.where(
            np.isin(groups, train_sessions)
        )[0]

        val_idx = np.where(
            np.isin(groups, val_sessions)
        )[0]

        test_idx = np.where(
            np.isin(groups, test_sessions)
        )[0]

        if (
            len(train_idx) == 0
            or len(val_idx) == 0
            or len(test_idx) == 0
        ):
            continue

        if not _has_minimum_samples(
            y,
            train_idx,
            MIN_TRAIN_SAMPLES_PER_CLASS
        ):
            continue

        if not _has_minimum_samples(
            y,
            val_idx,
            MIN_EVAL_SAMPLES_PER_CLASS
        ):
            continue

        if not _has_minimum_samples(
            y,
            test_idx,
            MIN_EVAL_SAMPLES_PER_CLASS
        ):
            continue

        # 동일 세션이 여러 분할에 포함되었는지 검사
        train_set = set(groups[train_idx])
        val_set = set(groups[val_idx])
        test_set = set(groups[test_idx])

        if (
            train_set & val_set
            or train_set & test_set
            or val_set & test_set
        ):

            raise RuntimeError(
                "데이터 분할 중 세션 중복이 발견되었습니다."
            )

        return (
            train_idx,
            val_idx,
            test_idx
        )

    raise RuntimeError(

        "세션 단위 분할에 실패했습니다. "

        "각 세션마다 STOP, UP, DOWN, OTHER를 모두 수집하고, "

        f"검증/테스트 각 클래스는 최소 "
        f"{MIN_EVAL_SAMPLES_PER_CLASS}개, "

        f"학습 각 클래스는 최소 "
        f"{MIN_TRAIN_SAMPLES_PER_CLASS}개를 확보하세요."
    )


# ============================================================
# 7. 예측 함수
# ============================================================

def predict(model, loader):

    actual = []
    predictions = []

    model.eval()

    with torch.no_grad():

        for batch_x, batch_y in loader:

            logits = model(
                batch_x.to(DEVICE)
            )

            predictions.extend(
                logits.argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

            actual.extend(
                batch_y.numpy().tolist()
            )

    return (
        np.asarray(actual),
        np.asarray(predictions)
    )


# ============================================================
# 8. 추론 속도 측정
# ============================================================

def measure_inference_ms(
    model,
    num_features
):

    dummy = torch.randn(
        1,
        SEQUENCE_LENGTH,
        num_features,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():

        for _ in range(15):
            model(dummy)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        elapsed_times = []

        for _ in range(100):

            start = time.perf_counter()

            model(dummy)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            elapsed_times.append(
                (time.perf_counter() - start) * 1000
            )

    return float(
        np.median(elapsed_times)
    )


# ============================================================
# 9. 단일 모델 학습 및 평가
# ============================================================

def train_one(
    extractor_name,
    x,
    y,
    groups,
    split,
    num_features
):

    train_idx, val_idx, test_idx = split

    x_train = x[train_idx]
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    x_test = x[test_idx]
    y_test = y[test_idx]

    train_loader = DataLoader(

        GestureDataset(
            x_train,
            y_train
        ),

        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(

        GestureDataset(
            x_val,
            y_val
        ),

        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(

        GestureDataset(
            x_test,
            y_test
        ),

        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE == "cuda")
    )

    model = GestureNet(
        num_features
    ).to(DEVICE)

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-4
    )

    # 클래스 불균형을 보정하기 위한 가중치
    train_counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES
    ).astype(np.float32)

    class_weights = (
        len(y_train)
        /
        (
            NUM_CLASSES
            * np.maximum(train_counts, 1.0)
        )
    )

    criterion = nn.CrossEntropyLoss(

        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    best_val_macro_f1 = -1.0

    best_epoch = 0

    best_state = None

    patience_left = PATIENCE

    train_sessions = sorted(
        np.unique(groups[train_idx]).tolist()
    )

    val_sessions = sorted(
        np.unique(groups[val_idx]).tolist()
    )

    test_sessions = sorted(
        np.unique(groups[test_idx]).tolist()
    )

    print("데이터 분할 결과")

    print(
        f"Train 세션: {train_sessions} "
        f"/ 샘플 {len(train_idx)}"
    )

    print(
        f"Validation 세션: {val_sessions} "
        f"/ 샘플 {len(val_idx)}"
    )

    print(
        f"Test 세션: {test_sessions} "
        f"/ 샘플 {len(test_idx)}"
    )

    # 학습
    for epoch in range(EPOCHS):

        model.train()

        losses = []

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(DEVICE)

            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        # Validation 평가
        val_actual, val_predictions = predict(
            model,
            val_loader
        )

        val_macro_f1 = f1_score(

            val_actual,

            val_predictions,

            average="macro",

            zero_division=0
        )

        if epoch == 0 or (epoch + 1) % 5 == 0:

            print(

                f"[{extractor_name}] "

                f"Epoch {epoch + 1:02d}/{EPOCHS} "

                f"| loss={np.mean(losses):.4f} "

                f"| Validation Macro F1={val_macro_f1:.4f}"
            )

        # Validation Macro F1 기준 최적 모델 저장
        if val_macro_f1 > best_val_macro_f1 + 1e-5:

            best_val_macro_f1 = float(
                val_macro_f1
            )

            best_epoch = epoch + 1

            best_state = {

                key: value.detach().cpu().clone()

                for key, value in model.state_dict().items()
            }

            patience_left = PATIENCE

        else:

            patience_left -= 1

            if patience_left <= 0:

                print(
                    "조기 종료: Validation Macro F1이 "
                    f"{PATIENCE} epoch 동안 개선되지 않았습니다."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "Validation 기준 최적 모델을 저장하지 못했습니다."
        )

    model.load_state_dict(best_state)

    model.eval()

    # 최적 모델 저장
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    weight_path = (
        MODEL_DIR
        / f"{extractor_name}_lstm_best.pth"
    )

    torch.save(
        model.state_dict(),
        weight_path
    )

    # 중요: Test는 학습 및 모델 선택에 사용하지 않은 데이터
    actual, predictions = predict(
        model,
        test_loader
    )

    labels = list(range(NUM_CLASSES))

    target_names = [
        CLASS_NAMES[label]
        for label in labels
    ]

    # 실제 평가 지표
    test_accuracy = float(
        accuracy_score(
            actual,
            predictions
        )
    )

    test_precision_macro = float(
        precision_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    test_recall_macro = float(
        recall_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    test_macro_f1 = float(
        f1_score(
            actual,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    test_weighted_f1 = float(
        f1_score(
            actual,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    test_balanced_accuracy = float(
        balanced_accuracy_score(
            actual,
            predictions
        )
    )

    # 클래스별 성능
    report = classification_report(

        actual,

        predictions,

        labels=labels,

        target_names=target_names,

        output_dict=True,

        zero_division=0
    )

    matrix = confusion_matrix(

        actual,

        predictions,

        labels=labels
    )

    inference_ms = measure_inference_ms(
        model,
        num_features
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 혼동 행렬 시각화
    plt.figure(figsize=(7, 5.5))

    sns.heatmap(

        matrix,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=target_names,

        yticklabels=target_names
    )

    plt.xlabel("Predicted label")

    plt.ylabel("True label")

    plt.title(
        f"{EXTRACTORS[extractor_name]['name']} "
        "+ LSTM (Independent Test Sessions)"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR
        / f"{extractor_name}_confusion_matrix.png",

        dpi=160
    )

    plt.close()

    true_counts = {

        CLASS_NAMES[label]: int(
            np.sum(actual == label)
        )

        for label in labels
    }

    predicted_counts = {

        CLASS_NAMES[label]: int(
            np.sum(predictions == label)
        )

        for label in labels
    }

    majority_baseline = float(
        max(true_counts.values())
        / max(len(actual), 1)
    )

    # 상세 보고서
    detail = {

        "extractor": extractor_name,

        "extractor_name": EXTRACTORS[
            extractor_name
        ]["name"],

        "evaluation_protocol": "session_independent_holdout",

        "best_epoch_selected_by_validation_macro_f1": best_epoch,

        "best_validation_macro_f1": best_val_macro_f1,

        "test_accuracy": test_accuracy,

        "test_macro_precision": test_precision_macro,

        "test_macro_recall": test_recall_macro,

        "test_macro_f1": test_macro_f1,

        "test_weighted_f1": test_weighted_f1,

        "test_balanced_accuracy": test_balanced_accuracy,

        "majority_class_baseline_accuracy": majority_baseline,

        "model_only_inference_ms_per_sequence": inference_ms,

        "train_sample_count": int(len(train_idx)),

        "validation_sample_count": int(len(val_idx)),

        "test_sample_count": int(len(test_idx)),

        "train_sessions": train_sessions,

        "validation_sessions": val_sessions,

        "test_sessions": test_sessions,

        "test_true_class_counts": true_counts,

        "test_predicted_class_counts": predicted_counts,

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),
    }

    report_path = (
        OUTPUT_DIR
        / f"{extractor_name}_report.json"
    )

    report_path.write_text(

        json.dumps(
            detail,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"
    )

    return {

        "extractor": extractor_name,

        "name": EXTRACTORS[
            extractor_name
        ]["name"],

        "validation_macro_f1": best_val_macro_f1,

        "test_accuracy": test_accuracy,

        "test_macro_precision": test_precision_macro,

        "test_macro_recall": test_recall_macro,

        "test_macro_f1": test_macro_f1,

        "test_weighted_f1": test_weighted_f1,

        "test_balanced_accuracy": test_balanced_accuracy,

        "majority_baseline_accuracy": majority_baseline,

        "model_inference_ms": round(inference_ms, 3),

        "train_samples": len(train_idx),

        "validation_samples": len(val_idx),

        "test_samples": len(test_idx),

        "session_count": len(np.unique(groups)),

        "best_epoch": best_epoch,
    }


# ============================================================
# 10. 전체 모델 학습 및 비교
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="산업현장 수신호 LSTM 학습 및 독립 테스트 평가"
    )

    parser.add_argument(

        "--extractor",

        choices=list(EXTRACTORS.keys()),

        default=None,

        help="특정 추출기만 학습"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    set_seed(args.seed)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    extractors = (
        [args.extractor]
        if args.extractor
        else list(EXTRACTORS.keys())
    )

    results = []

    for extractor_name in extractors:

        print("\n" + "=" * 68)

        print(
            f"{EXTRACTORS[extractor_name]['name']} "
            "+ LSTM 학습 및 독립 테스트"
        )

        print("=" * 68)

        try:

            x, y, groups, num_features, _ = load_data(
                extractor_name
            )

            session_ids = np.unique(groups)

            total_counts = {

                CLASS_NAMES[label]: int(
                    np.sum(y == label)
                )

                for label in range(NUM_CLASSES)
            }

            print(f"전체 샘플 수: {len(y)}")

            print(f"독립 세션 수: {len(session_ids)}")

            print(f"클래스별 샘플 수: {total_counts}")

            for session_id in session_ids:

                counts = {

                    CLASS_NAMES[label]: int(
                        np.sum(
                            (groups == session_id)
                            & (y == label)
                        )
                    )

                    for label in range(NUM_CLASSES)
                }

                print(
                    f"세션 {session_id}: {counts}"
                )

            if min(total_counts.values()) < MIN_TRAIN_SAMPLES_PER_CLASS:

                raise RuntimeError(
                    "일부 클래스의 전체 샘플이 부족합니다. "
                    "STOP, UP, DOWN, OTHER를 더 수집하세요."
                )

            split = split_by_session(
                x,
                y,
                groups,
                seed=args.seed
            )

            result = train_one(

                extractor_name,

                x,

                y,

                groups,

                split,

                num_features
            )

            results.append(result)

            print(

                "최종 독립 TEST 결과 | "

                f"Accuracy={result['test_accuracy']:.4f} | "

                f"Macro Precision={result['test_macro_precision']:.4f} | "

                f"Macro Recall={result['test_macro_recall']:.4f} | "

                f"Macro F1={result['test_macro_f1']:.4f} | "

                f"Balanced Accuracy={result['test_balanced_accuracy']:.4f}"
            )

        except Exception as exc:

            print(
                f"[학습/평가 중단] {extractor_name}: {exc}"
            )

    if not results:

        print(
            "\n학습 가능한 모델이 없습니다. "
            "오류 메시지와 데이터 폴더를 확인하세요."
        )

        return

    comparison = pd.DataFrame(results)

    comparison_path = (
        OUTPUT_DIR / "model_comparison.csv"
    )

    comparison.to_csv(

        comparison_path,

        index=False,

        encoding="utf-8-sig"
    )

    # 모델 비교 그래프
    metric_columns = [

        ("test_accuracy", "Test Accuracy"),

        ("test_macro_f1", "Test Macro F1"),

        ("test_balanced_accuracy", "Test Balanced Accuracy"),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    bar_colors = [
        "#4169A1",
        "#4F9D75",
        "#C66B5D"
    ]

    for axis, (column, title) in zip(
        axes,
        metric_columns
    ):

        bars = axis.bar(

            comparison["name"],

            comparison[column],

            color=bar_colors[:len(comparison)]
        )

        axis.set_title(title)

        axis.set_ylim(0, 1.05)

        axis.set_ylabel("Score")

        axis.tick_params(
            axis="x",
            rotation=15
        )

        for bar, value in zip(
            bars,
            comparison[column]
        ):

            axis.text(

                bar.get_x() + bar.get_width() / 2,

                min(float(value) + 0.02, 1.03),

                f"{value:.3f}",

                ha="center",

                va="bottom"
            )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_comparison.png",
        dpi=160
    )

    plt.close(figure)

    # 모델 선택은 Test가 아닌 Validation Macro F1 기준
    best_row = comparison.sort_values(

        "validation_macro_f1",

        ascending=False
    ).iloc[0]

    (MODEL_DIR / "best_extractor.txt").write_text(

        str(best_row["extractor"]),

        encoding="utf-8"
    )

    print("\n" + "=" * 68)

    print("독립 세션 TEST 모델 성능 비교")

    print("=" * 68)

    print(comparison.to_string(index=False))

    print(
        f"\nValidation 기준 선택 모델: {best_row['name']}"
    )

    print("결과 파일:")

    print(f"- {comparison_path}")

    print(f"- {OUTPUT_DIR / '*_report.json'}")

    print(f"- {OUTPUT_DIR / '*_confusion_matrix.png'}")

    print(f"- {OUTPUT_DIR / 'model_comparison.png'}")


if __name__ == "__main__":

    main()
