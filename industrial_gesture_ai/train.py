"""
3가지 관절 추출기 × LSTM 모델 학습 및 성능 비교.

각 extractor별로 별도의 데이터 폴더(data/yolo, data/hands, data/holistic)에서 로드하여
동일한 LSTM 구조로 학습한 뒤 Macro F1 기준으로 비교한다.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, SEQUENCE_LENGTH, EXTRACTORS, NUM_CLASSES
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
HIDDEN = 96


class GestureDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GestureNet(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN, classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, num_layers=2,
            batch_first=True, dropout=0.2
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def load_data(extractor: str):
    num_features = EXTRACTORS[extractor]["num_features"]
    data_root = DATA_DIR / extractor
    xs, ys = [], []

    for label, name in CLASS_NAMES.items():
        folder = data_root / name
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            arr = np.load(file)
            if arr.shape == (SEQUENCE_LENGTH, num_features):
                xs.append(arr)
                ys.append(label)

    if not xs:
        raise RuntimeError(f"[{extractor}] 학습 데이터가 없습니다. 먼저 collect_data.py --extractor {extractor} 를 실행하세요.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), num_features


def train_one(extractor: str, x_train, y_train, x_val, y_val, num_features: int):
    model = GestureNet(num_features).to(DEVICE)
    train_loader = DataLoader(GestureDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(GestureDataset(x_val, y_val), batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        preds, actual = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                actual.extend(yb.numpy())

        val_f1 = f1_score(actual, preds, average="macro", zero_division=0)
        val_acc = accuracy_score(actual, preds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{extractor}] Epoch {epoch+1:02d}/{EPOCHS} | loss={np.mean(losses):.4f} | val_acc={val_acc:.4f} | macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_DIR / f"{extractor}_lstm_best.pth")

    # Final evaluation with best weights + inference speed
    model.load_state_dict(torch.load(MODEL_DIR / f"{extractor}_lstm_best.pth", map_location=DEVICE))
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            actual.extend(yb.numpy())

    # 추론 속도 측정 (ms / sample)
    dummy = torch.randn(1, SEQUENCE_LENGTH, num_features).to(DEVICE)
    # warmup
    for _ in range(10):
        _ = model(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    N = 100
    with torch.no_grad():
        for _ in range(N):
            _ = model(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / N * 1000

    report = classification_report(
        actual, preds, labels=list(range(NUM_CLASSES)),
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(actual, preds, labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{EXTRACTORS[extractor]['name']} + LSTM")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{extractor}_confusion_matrix.png", dpi=150)
    plt.close()

    with open(OUTPUT_DIR / f"{extractor}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "extractor": extractor,
        "name": EXTRACTORS[extractor]["name"],
        "accuracy": accuracy_score(actual, preds),
        "macro_f1": f1_score(actual, preds, average="macro", zero_division=0),
        "inference_ms": round(elapsed_ms, 3),
        "num_samples": len(y_train) + len(y_val),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []
    for extractor in EXTRACTORS.keys():
        print(f"\n{'='*60}")
        print(f"학습 시작: {EXTRACTORS[extractor]['name']} + LSTM")
        print(f"{'='*60}")
        try:
            x, y, num_features = load_data(extractor)
            counts = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(NUM_CLASSES)}
            print(f"데이터 수: {len(y)} | {counts}")
            if min(counts.values()) < 8:
                print(f"  → 클래스당 샘플이 너무 적습니다. 건너뜁니다.")
                continue

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )
            res = train_one(extractor, x_train, y_train, x_val, y_val, num_features)
            results.append(res)
            print(f"  → 최종 Macro F1: {res['macro_f1']:.4f}")
        except Exception as e:
            print(f"  → 실패: {e}")

    if not results:
        print("\n학습된 모델이 없습니다. 데이터를 먼저 수집하세요.")
        return

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n" + "="*60)
    print("모델 성능 비교")
    print("="*60)
    print(df.to_string(index=False))

    # 시각화: Accuracy, Macro F1, Inference Time
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    axes[0].bar(df["name"], df["accuracy"], color=colors[:len(df)])
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    for i, v in enumerate(df["accuracy"]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

    axes[1].bar(df["name"], df["macro_f1"], color=colors[:len(df)])
    axes[1].set_title("Macro F1")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(df["macro_f1"]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center")

    axes[2].bar(df["name"], df["inference_ms"], color=colors[:len(df)])
    axes[2].set_title("Inference Time (ms / sample)")
    axes[2].set_ylabel("ms")
    for i, v in enumerate(df["inference_ms"]):
        axes[2].text(i, v + 0.05, f"{v:.2f}", ha="center")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"\n비교 그래프 저장: {OUTPUT_DIR / 'model_comparison.png'}")

    best = df.iloc[0]["extractor"]
    (MODEL_DIR / "best_extractor.txt").write_text(best, encoding="utf-8")
    print(f"\n최종 선택 모델: {EXTRACTORS[best]['name']} + LSTM")
    print(f"  - Accuracy: {df.iloc[0]['accuracy']:.4f}")
    print(f"  - Macro F1: {df.iloc[0]['macro_f1']:.4f}")
    print(f"  - Inference: {df.iloc[0]['inference_ms']:.2f} ms")


if __name__ == "__main__":
    main()
