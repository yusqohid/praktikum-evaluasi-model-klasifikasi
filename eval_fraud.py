"""Praktikum Data Mining 2026 - Evaluasi Model Klasifikasi.

Skrip ini membuat dataset transaksi pembayaran digital sintetis, melatih
model Decision Tree, lalu menghitung confusion matrix dan metrik evaluasi
(accuracy, precision, recall, F1-score) beserta eksperimen baseline dan
ambang prediksi.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def cetak_judul(judul: str) -> None:
    print("\n" + "=" * 68)
    print(judul)
    print("=" * 68)


# 4.1 Membuat dataset sintetis
cetak_judul("4.1 Dataset sintetis transaksi pembayaran digital")
rng = np.random.default_rng(RANDOM_STATE)
n = 500

amount = rng.lognormal(mean=5.0, sigma=1.0, size=n)
hour = rng.integers(0, 24, n)
new_device = rng.binomial(1, 0.25, n)
foreign_ip = rng.binomial(1, 0.12, n)

score = (
    -4.5
    + 0.45 * (amount > 300)
    + 1.1 * new_device
    + 1.5 * foreign_ip
    + 0.7 * (hour < 5)
)
prob = 1 / (1 + np.exp(-score))
isFraud = rng.binomial(1, prob)

df = pd.DataFrame(
    {
        "amount": amount.round(2),
        "hour": hour,
        "new_device": new_device,
        "foreign_ip": foreign_ip,
        "isFraud": isFraud,
    }
)

print(df.head().to_string(index=False))
print("\nDistribusi kelas:")
counts = df["isFraud"].value_counts().sort_index()
print(counts.to_string())
prop = df["isFraud"].mean()
print(f"\nTotal transaksi : {len(df)}")
print(f"Fraud           : {counts.get(1, 0)} ({prop * 100:.2f}%)")
print(f"Normal          : {counts.get(0, 0)} ({(1 - prop) * 100:.2f}%)")

# 4.2 Membagi data dan melatih model
cetak_judul("4.2 Split data dan latih Decision Tree")
X = df.drop(columns="isFraud")
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

model = DecisionTreeClassifier(
    max_depth=4, class_weight="balanced", random_state=RANDOM_STATE
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Data latih : {len(y_train)}")
print(f"Data uji   : {len(y_test)}")
print(f"Fraud di data uji: {int(y_test.sum())} ({y_test.mean() * 100:.2f}%)")
print(f"Kedalaman pohon: {model.get_depth()}, jumlah leaf: {model.get_n_leaves()}")

# 4.3 Confusion matrix
cetak_judul("4.3 Confusion matrix")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix:")
print(cm)
print(f"\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")

# 4.4 Perhitungan manual dan perbandingan dengan scikit-learn
cetak_judul("4.4 Perhitungan manual dan verifikasi scikit-learn")
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

print("Manual :")
print(f"  Accuracy  = ({tp}+{tn})/{tp + tn + fp + fn} = {accuracy:.4f}")
print(f"  Precision = {tp}/({tp}+{fp}) = {precision:.4f}")
print(f"  Recall    = {tp}/({tp}+{fn}) = {recall:.4f}")
print(f"  F1-score  = {f1:.4f}")

skor = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
}
print("\nscikit-learn :")
for nama, nilai in skor.items():
    print(f"  {nama:<10} = {nilai:.4f}")
print("\nSelisih manual vs sklearn:")
print(
    f"  accuracy={abs(accuracy - skor['accuracy']):.2e}, "
    f"precision={abs(precision - skor['precision']):.2e}, "
    f"recall={abs(recall - skor['recall']):.2e}, "
    f"f1={abs(f1 - skor['f1']):.2e}"
)

# 4.5 Visualisasi confusion matrix
cetak_judul("4.5 Visualisasi confusion matrix")
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"]).plot(
    cmap="Blues", values_format="d", ax=ax, colorbar=False
)
ax.set_title("Confusion Matrix - Deteksi Fraud")
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
fig.tight_layout()
cm_path = FIG_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=160)
plt.close(fig)
print(f"Gambar disimpan: {cm_path}")

# 4.7 Baseline semua transaksi diprediksi normal
cetak_judul("4.7 Eksperimen baseline: semua transaksi diprediksi normal")
baseline = np.zeros(len(y_test), dtype=int)
baseline_acc = accuracy_score(y_test, baseline)
print(f"Baseline accuracy: {baseline_acc:.4f}")
print(f"Baseline recall fraud: {recall_score(y_test, baseline, zero_division=0):.4f}")
print("\nclassification_report baseline (labels=[0,1]):")
print(classification_report(y_test, baseline, labels=[0, 1], zero_division=0))
print("classification_report Decision Tree (labels=[0,1]):")
print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

# 4.8 Pengaruh ambang prediksi
cetak_judul("4.8 Pengaruh ambang prediksi")
y_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

rows = []
for threshold in [0.2, 0.4, 0.5, 0.7]:
    pred = (y_prob >= threshold).astype(int)
    cm_t = confusion_matrix(y_test, pred, labels=[0, 1])
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    rows.append(
        {
            "threshold": threshold,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "tp": int(tp_t),
            "fp": int(fp_t),
            "fn": int(fn_t),
            "tn": int(tn_t),
        }
    )

thr_df = pd.DataFrame(rows)
print(thr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

uniq = np.unique(np.round(y_prob, 4))
print(f"\nProbabilitas unik dari leaf pohon ({len(uniq)}): {uniq.tolist()}")

# Tugas 1 Hitung manual
cetak_judul("TUGAS 1 - Hitung manual dari confusion matrix yang diberikan")
t1 = {"tp": 24, "tn": 156, "fp": 12, "fn": 8}
total1 = sum(t1.values())
acc1 = (t1["tp"] + t1["tn"]) / total1
prec1 = t1["tp"] / (t1["tp"] + t1["fp"])
rec1 = t1["tp"] / (t1["tp"] + t1["fn"])
f1_1 = 2 * prec1 * rec1 / (prec1 + rec1)
print(f"Confusion matrix: [[TN={t1['tn']}, FP={t1['fp']}], [FN={t1['fn']}, TP={t1['tp']}]]")
print(f"Total observasi : {total1}")
print(f"Accuracy  = ({t1['tp']}+{t1['tn']})/{total1} = {acc1:.4f}")
print(f"Precision = 24/36 = {prec1:.4f}")
print(f"Recall    = 24/32 = {rec1:.4f}")
print(f"F1-score  = {f1_1:.4f}")
