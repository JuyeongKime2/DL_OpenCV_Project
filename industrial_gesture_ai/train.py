"""세션 독립 분할 기반 LSTM 학습 및 최종 테스트 평가 (데이터 증강 및 규제 추가)."""

import argparse
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import platform

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

# ============================================================
# [튜닝 포인트 1] 하이퍼파라미터 조정 (과적합 방지)
# ============================================================
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 5e-4  # 학습률 하향 조정 (기존 1e-3 -> 5e-4)
HIDDEN_SIZE = 64      # 은닉층 크기 축소 (기존 96 -> 64)
PATIENCE = 10
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# [튜닝 포인트 2] 데이터셋 및 실시간 데이터 증강(Augmentation)
# ============================================================
class GestureDataset(Dataset):
    def __init__(self, x, y, augment=False):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sequence = self.x[index].clone()
        
        # 학습 시에만 노이즈 및 크기 변환 적용
        if self.augment:
            # Jittering (가우시안 노이즈 추가)
            noise = torch.randn_like(sequence) * 0.015
            sequence = sequence + noise
            
            # Scaling (크기 조절 0.9 ~ 1.1배)
            scale_factor = torch.empty(1).uniform_(0.9, 1.1).item()
            sequence = sequence * scale_factor

        return sequence, self.y[index]


# ============================================================
# [튜닝 포인트 3] LSTM 모델 규제 강화 (Dropout & 단순화)
# ============================================================
class GestureNet(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN_SIZE, classes=NUM_CLASSES):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=1,         # 레이어 수 축소 (기존 2 -> 1)
            batch_first=True,
            # num_layers=1일 경우 lstm 내부 dropout은 무시되므로 제거
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),      # Dropout 비율 강화 (기존 0.25 -> 0.4)
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


# ============================================================
# 4. 세션 단위 데이터 로드
# ============================================================
def load_data(extractor_name):
    num_features = EXTRACTORS[extractor_name]["num_features"]
    data_root = DATA_DIR / extractor_name
    sessions_root = data_root / "sessions"
    legacy_files = []

    if data_root.exists():
        for class_name in CLASS_NAMES.values():
            legacy_files.extend((data_root / class_name).glob("*.npy"))

    if legacy_files:
        print(f"[안내] 기존 flat 형식 데이터 {len(legacy_files)}개를 발견했습니다. 세션 분할이 불가능하므로 학습에서 제외합니다.")

    if not sessions_root.exists():
        raise RuntimeError(f"세션 데이터 폴더가 없습니다: {sessions_root}\n먼저 python collect_data.py --extractor {extractor_name} 를 실행하세요.")

    x_list = []
    y_list = []
    group_list = []
    path_list = []
    session_dirs = sorted(path for path in sessions_root.iterdir() if path.is_dir())

    for session_dir in session_dirs:
        session_id = session_dir.name
        for label_id, class_name in CLASS_NAMES.items():
            class_dir = session_dir / class_name
            if not class_dir.exists():
                continue
            for file_path in sorted(class_dir.glob("*.npy")):
                try:
                    sample = np.load(file_path, allow_pickle=False)
                except Exception as exc:
                    print(f"[제외] 파일을 읽지 못함: {file_path} ({exc})")
                    continue
                if sample.shape != (SEQUENCE_LENGTH, num_features):
                    print(f"[제외] 데이터 크기 오류: {file_path.name}, 현재={sample.shape}, 기대={(SEQUENCE_LENGTH, num_features)}")
                    continue
                if not np.isfinite(sample).all():
                    print(f"[제외] NaN/Inf 포함: {file_path}")
                    continue
                
                x_list.append(sample.astype(np.float32))
                y_list.append(label_id)
                group_list.append(session_id)
                path_list.append(str(file_path))

    if not x_list:
        raise RuntimeError("학습 가능한 세션 데이터가 없습니다. 데이터 수집을 먼저 진행하세요.")

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    groups = np.asarray(group_list)
    return x, y, groups, num_features, path_list


# ============================================================
# 5. 세션 분할 보조 함수
# ============================================================
def _class_counts(y, indices):
    return np.bincount(y[indices], minlength=NUM_CLASSES)

def _has_minimum_samples(y, indices, minimum):
    return bool(np.all(_class_counts(y, indices) >= minimum))


# ============================================================
# 6. 세션 단위 Train / Validation / Test 분할
# ============================================================
def split_by_session(x, y, groups, seed=RANDOM_SEED):
    unique_sessions = np.unique(groups)
    session_count = len(unique_sessions)

    if session_count < max(MIN_SESSION_COUNT, 2):
        print(f"[경고] 독립 세션이 {session_count}개입니다.")
        print("        세션 독립 평가가 불가능하여 계층적 랜덤 분할을 사용합니다.")
        print("        성능 수치는 실제보다 높게 나올 수 있습니다.")
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=y, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=seed)
        return train_idx, val_idx, test_idx

    test_session_count = max(1, int(round(session_count * 0.20)))
    val_session_count = max(1, int(round(session_count * 0.20)))
    if test_session_count + val_session_count >= session_count:
        test_session_count = 1
        val_session_count = 1

    for attempt in range(2000):
        rng = np.random.default_rng(seed + attempt)
        shuffled = rng.permutation(unique_sessions)
        
        test_sessions = shuffled[:test_session_count]
        val_sessions = shuffled[test_session_count:test_session_count + val_session_count]
        train_sessions = shuffled[test_session_count + val_session_count:]
        
        train_idx = np.where(np.isin(groups, train_sessions))[0]
        val_idx = np.where(np.isin(groups, val_sessions))[0]
        test_idx = np.where(np.isin(groups, test_sessions))[0]

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0: continue
        if not _has_minimum_samples(y, train_idx, MIN_TRAIN_SAMPLES_PER_CLASS): continue
        if not _has_minimum_samples(y, val_idx, MIN_EVAL_SAMPLES_PER_CLASS): continue
        if not _has_minimum_samples(y, test_idx, MIN_EVAL_SAMPLES_PER_CLASS): continue

        train_set = set(groups[train_idx])
        val_set = set(groups[val_idx])
        test_set = set(groups[test_idx])

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError("데이터 분할 중 세션 중복이 발견되었습니다.")
        
        return train_idx, val_idx, test_idx

    raise RuntimeError(
        "세션 단위 분할에 실패했습니다. 각 세션마다 STOP, UP, DOWN, OTHER를 모두 수집하고, "
        f"검증/테스트 각 클래스는 최소 {MIN_EVAL_SAMPLES_PER_CLASS}개, 학습 각 클래스는 최소 {MIN_TRAIN_SAMPLES_PER_CLASS}개를 확보하세요."
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
            logits = model(batch_x.to(DEVICE))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            actual.extend(batch_y.numpy().tolist())
    return np.asarray(actual), np.asarray(predictions)


# ============================================================
# 8. 추론 속도 측정
# ============================================================
def measure_inference_ms(model, num_features):
    dummy = torch.randn(1, SEQUENCE_LENGTH, num_features, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(15): model(dummy)
        if DEVICE == "cuda": torch.cuda.synchronize()
        elapsed_times = []
        for _ in range(100):
            start = time.perf_counter()
            model(dummy)
            if DEVICE == "cuda": torch.cuda.synchronize()
            elapsed_times.append((time.perf_counter() - start) * 1000)
    return float(np.median(elapsed_times))


# ============================================================
# 9. 단일 모델 학습 및 평가
# ============================================================
def train_one(extractor_name, x, y, groups, split, num_features):
    train_idx, val_idx, test_idx = split
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    train_loader = DataLoader(
        GestureDataset(x_train, y_train, augment=True), # 학습 시에만 Augment 활성화
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        GestureDataset(x_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(
        GestureDataset(x_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=(DEVICE == "cuda")
    )

    model = GestureNet(num_features).to(DEVICE)
    
    # [튜닝 포인트 4] L2 Regularization(가중치 감쇠) 강화
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-3 # 기존 1e-4 -> 1e-3 변경
    )

    train_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = len(y_train) / (NUM_CLASSES * np.maximum(train_counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_left = PATIENCE

    train_sessions = sorted(np.unique(groups[train_idx]).tolist())
    val_sessions = sorted(np.unique(groups[val_idx]).tolist())
    test_sessions = sorted(np.unique(groups[test_idx]).tolist())

    print("데이터 분할 결과")
    print(f"Train 세션: {train_sessions} / 샘플 {len(train_idx)}")
    print(f"Validation 세션: {val_sessions} / 샘플 {len(val_idx)}")
    print(f"Test 세션: {test_sessions} / 샘플 {len(test_idx)}")

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_actual, val_predictions = predict(model, val_loader)
        val_macro_f1 = f1_score(val_actual, val_predictions, average="macro", zero_division=0)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"[{extractor_name}] Epoch {epoch + 1:02d}/{EPOCHS} | loss={np.mean(losses):.4f} | Validation Macro F1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"조기 종료: Validation Macro F1이 {PATIENCE} epoch 동안 개선되지 않았습니다.")
                break

    if best_state is None:
        raise RuntimeError("Validation 기준 최적 모델을 저장하지 못했습니다.")

    model.load_state_dict(best_state)
    model.eval()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = MODEL_DIR / f"{extractor_name}_lstm_best.pth"
    torch.save(model.state_dict(), weight_path)

    actual, predictions = predict(model, test_loader)
    labels = list(range(NUM_CLASSES))
    target_names = [CLASS_NAMES[label] for label in labels]

    test_accuracy = float(accuracy_score(actual, predictions))
    test_precision_macro = float(precision_score(actual, predictions, average="macro", zero_division=0))
    test_recall_macro = float(recall_score(actual, predictions, average="macro", zero_division=0))
    test_macro_f1 = float(f1_score(actual, predictions, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(actual, predictions, average="weighted", zero_division=0))
    test_balanced_accuracy = float(balanced_accuracy_score(actual, predictions))

    report = classification_report(actual, predictions, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predictions, labels=labels)
    inference_ms = measure_inference_ms(model, num_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    protocol_note = "Single-Session Random Split (Optimistic)" if len(np.unique(groups)) < 2 else "Independent Test Sessions"
    plt.title(f"{EXTRACTORS[extractor_name]['name']} + LSTM ({protocol_note})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{extractor_name}_confusion_matrix.png", dpi=160)
    plt.close()

    true_counts = {CLASS_NAMES[label]: int(np.sum(actual == label)) for label in labels}
    predicted_counts = {CLASS_NAMES[label]: int(np.sum(predictions == label)) for label in labels}
    majority_baseline = float(max(true_counts.values()) / max(len(actual), 1))

    detail = {
        "extractor": extractor_name,
        "extractor_name": EXTRACTORS[extractor_name]["name"],
        "evaluation_protocol": "single_session_stratified_random_split" if len(np.unique(groups)) < 2 else "session_independent_holdout",
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

    report_path = OUTPUT_DIR / f"{extractor_name}_report.json"
    report_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "extractor": extractor_name,
        "name": EXTRACTORS[extractor_name]["name"],
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

# ============================================================
# 11. 화려한 시각화 (레이더, 속도-정확도, 히트맵, 대시보드)
# ============================================================
COLORS = {"yolo": "#4169A1", "hands": "#4F9D75", "holistic": "#C66B5D"}


def _load_reports():
    reports = {}
    for name in EXTRACTORS.keys():
        path = OUTPUT_DIR / f"{name}_report.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                reports[name] = json.load(f)
    return reports


def plot_radar(reports, save_path):
    metrics = ["test_accuracy", "test_macro_f1", "test_balanced_accuracy",
               "test_macro_precision", "test_macro_recall"]
    labels = ["Accuracy", "Macro F1", "Balanced Acc", "Precision", "Recall"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for name, report in reports.items():
        values = [report.get(m, 0) for m in metrics]
        values += values[:1]
        color = COLORS.get(name, "#888")
        ax.plot(angles, values, "o-", linewidth=2, label=EXTRACTORS[name]["name"], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("모델 성능 레이더 차트", size=16, pad=20, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_speed_accuracy(df, save_path):
    if df is None or "model_inference_ms" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        color = COLORS.get(row["extractor"], "#888")
        ax.scatter(row["model_inference_ms"], row["test_accuracy"], s=300,
                   color=color, edgecolors="black", linewidth=1.5, zorder=3)
        ax.annotate(row["name"], (row["model_inference_ms"], row["test_accuracy"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")
    ax.set_xlabel("Inference Time (ms / sequence)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("속도 vs 정확도 트레이드오프", fontsize=16, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_class_heatmap(reports, save_path):
    class_names = list(CLASS_NAMES.values())
    data = []
    for name, report in reports.items():
        cr = report.get("classification_report", {})
        row = [cr.get(cname, {}).get("f1-score", 0) for cname in class_names]
        data.append(row)
    df = pd.DataFrame(data, index=[EXTRACTORS[n]["name"] for n in reports.keys()], columns=class_names)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("클래스별 F1-Score 히트맵", fontsize=16, weight="bold", pad=12)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_dashboard(reports, df, save_path):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    metrics = ["test_accuracy", "test_macro_f1", "test_balanced_accuracy"]
    labels = ["Accuracy", "Macro F1", "Balanced Acc"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for name, report in reports.items():
        values = [report.get(m, 0) for m in metrics]
        values += values[:1]
        color = COLORS.get(name, "#888")
        ax1.plot(angles, values, "o-", linewidth=2, color=color, label=EXTRACTORS[name]["name"])
        ax1.fill(angles, values, alpha=0.12, color=color)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("핵심 지표 레이더", size=13, weight="bold", pad=15)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    if df is not None:
        names = df["name"].tolist()
        f1s = df["test_macro_f1"].tolist()
        colors = [COLORS.get(e, "#888") for e in df["extractor"]]
        bars = ax2.barh(names, f1s, color=colors, edgecolor="black")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Test Macro F1")
        ax2.set_title("Macro F1 비교", size=13, weight="bold")
        for bar, v in zip(bars, f1s):
            ax2.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 0])
    if df is not None and "model_inference_ms" in df.columns:
        names = df["name"].tolist()
        speeds = df["model_inference_ms"].tolist()
        colors = [COLORS.get(e, "#888") for e in df["extractor"]]
        bars = ax3.bar(names, speeds, color=colors, edgecolor="black")
        ax3.set_ylabel("Inference Time (ms)")
        ax3.set_title("추론 속도 비교", size=13, weight="bold")
        ax3.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, speeds):
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary = ["모델 성능 요약\n"]
    if df is not None:
        best = df.sort_values("test_macro_f1", ascending=False).iloc[0]
        summary.append(f"최고 Macro F1: {best['name']}")
        summary.append(f"  → {best['test_macro_f1']:.4f}")
        if "model_inference_ms" in df.columns:
            fastest = df.sort_values("model_inference_ms").iloc[0]
            summary.append(f"\n최고 속도: {fastest['name']}")
            summary.append(f"  → {fastest['model_inference_ms']:.2f} ms")
    protocols = {r.get("evaluation_protocol", "") for r in reports.values()}
    if "single_session_stratified_random_split" in protocols:
        summary.append("\n⚠ 평가 프로토콜:")
        summary.append("단일 세션 랜덤 분할")
        summary.append("(과대평가 가능성 있음)")
    ax4.text(0.05, 0.95, "\n".join(summary), transform=ax4.transAxes, fontsize=12,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6", pad=0.8))
    ax4.set_title("요약", size=13, weight="bold")

    fig.suptitle("산업현장 수신호 인식 모델 비교 대시보드", fontsize=18, weight="bold", y=0.98)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_fancy_visualizations(comparison_df):
    """학습 종료 후 화려한 시각화 일괄 생성."""
    reports = _load_reports()
    if not reports:
        print("[시각화] report.json이 없어 건너뜁니다.")
        return
    print("\n화려한 시각화 생성 중...")
    plot_radar(reports, OUTPUT_DIR / "radar_chart.png")
    plot_speed_accuracy(comparison_df, OUTPUT_DIR / "speed_accuracy.png")
    plot_class_heatmap(reports, OUTPUT_DIR / "class_f1_heatmap.png")
    plot_dashboard(reports, comparison_df, OUTPUT_DIR / "comparison_dashboard.png")
    print("시각화 완료:")
    print(f"  - {OUTPUT_DIR / 'radar_chart.png'}")
    print(f"  - {OUTPUT_DIR / 'speed_accuracy.png'}")
    print(f"  - {OUTPUT_DIR / 'class_f1_heatmap.png'}")
    print(f"  - {OUTPUT_DIR / 'comparison_dashboard.png'}")


def main():
    parser = argparse.ArgumentParser(description="산업현장 수신호 LSTM 학습 및 독립 테스트 평가")
    parser.add_argument("--extractor", choices=list(EXTRACTORS.keys()), default=None, help="특정 추출기만 학습")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extractors = [args.extractor] if args.extractor else list(EXTRACTORS.keys())
    results = []

    for extractor_name in extractors:
        print("\n" + "=" * 68)
        print(f"{EXTRACTORS[extractor_name]['name']} + LSTM 학습 및 독립 테스트")
        print("=" * 68)
        
        try:
            x, y, groups, num_features, _ = load_data(extractor_name)
            session_ids = np.unique(groups)
            total_counts = {CLASS_NAMES[label]: int(np.sum(y == label)) for label in range(NUM_CLASSES)}
            
            print(f"전체 샘플 수: {len(y)}")
            print(f"독립 세션 수: {len(session_ids)}")
            print(f"클래스별 샘플 수: {total_counts}")
            
            for session_id in session_ids:
                counts = {CLASS_NAMES[label]: int(np.sum((groups == session_id) & (y == label))) for label in range(NUM_CLASSES)}
                print(f"세션 {session_id}: {counts}")

            if min(total_counts.values()) < MIN_TRAIN_SAMPLES_PER_CLASS:
                raise RuntimeError("일부 클래스의 전체 샘플이 부족합니다. STOP, UP, DOWN, OTHER를 더 수집하세요.")

            split = split_by_session(x, y, groups, seed=args.seed)
            result = train_one(extractor_name, x, y, groups, split, num_features)
            results.append(result)
            
            print(
                "최종 TEST 결과 | "
                f"Accuracy={result['test_accuracy']:.4f} | "
                f"Macro Precision={result['test_macro_precision']:.4f} | "
                f"Macro Recall={result['test_macro_recall']:.4f} | "
                f"Macro F1={result['test_macro_f1']:.4f} | "
                f"Balanced Accuracy={result['test_balanced_accuracy']:.4f}"
            )

        except Exception as exc:
            print(f"[학습/평가 중단] {extractor_name}: {exc}")

    if not results:
        print("\n학습 가능한 모델이 없습니다. 오류 메시지와 데이터 폴더를 확인하세요.")
        return

    comparison = pd.DataFrame(results)
    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    metric_columns = [
        ("test_accuracy", "Test Accuracy"),
        ("test_macro_f1", "Test Macro F1"),
        ("test_balanced_accuracy", "Test Balanced Accuracy"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    bar_colors = ["#4169A1", "#4F9D75", "#C66B5D"]

    for axis, (column, title) in zip(axes, metric_columns):
        bars = axis.bar(comparison["name"], comparison[column], color=bar_colors[:len(comparison)])
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", rotation=15)
        
        for bar, value in zip(bars, comparison[column]):
            axis.text(bar.get_x() + bar.get_width() / 2, min(float(value) + 0.02, 1.03), f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)

    best_row = comparison.sort_values("validation_macro_f1", ascending=False).iloc[0]
    (MODEL_DIR / "best_extractor.txt").write_text(str(best_row["extractor"]), encoding="utf-8")

    print("\n" + "=" * 68)
    if any(r.get("session_count", 0) < 2 for r in results):
        print("모델 성능 비교 (단일 세션 랜덤 분할 - 과대평가 가능)")
    else:
        print("독립 세션 TEST 모델 성능 비교")
    print("=" * 68)
    print(comparison.to_string(index=False))
    print(f"\nValidation 기준 선택 모델: {best_row['name']}")
    print("결과 파일:")
    print(f"- {comparison_path}")
    print(f"- {OUTPUT_DIR / '*_report.json'}")
    print(f"- {OUTPUT_DIR / '*_confusion_matrix.png'}")
    print(f"- {OUTPUT_DIR / 'model_comparison.png'}")



    # 화려한 시각화 생성
    generate_fancy_visualizations(comparison)

if __name__ == "__main__":
    main()