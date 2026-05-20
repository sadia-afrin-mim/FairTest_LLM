# -*- coding: utf-8 -*-
"""
retrain_and_compare.py
Retrains the model using discriminatory samples found by FairAgent
and compares SPD + accuracy before and after retraining.

Dataset         : Bank Marketing (bank-full-biased.csv)
Protected attr  : marital
Label column    : y  (0=no, 1=yes)

CORRECT approach:
  1. Load FULL bank dataset
  2. Split into train/test FIRST (test set LOCKED)
  3. Train original model on train split
  4. Load discriminatory pairs from FairAgent
  5. Combine train split + discriminatory pairs
  6. Retrain on combined
  7. Evaluate BOTH on same locked test split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

# ── SETTINGS ───────────────────────────────────────────────────
DATASET_PATH   = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bank-full-biased.csv')   # ← full bank dataset
DISC_CSV       = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', 'bank', '3', 'local_samples_MLP_expga.csv')
OUTPUT_DIR     = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', 'bank', '3')

PROTECTED_COL  = 'marital'
LABEL_COL      = 'y'
PRIV_VAL       = 1   # married   (privileged)
UNPRIV_VAL     = 2   # single    (unprivileged)

TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# Bank dataset column names (no header in file)
BANK_COLS = ['age','job','marital','education','default',
             'balance','housing','loan','contact','day',
             'month','duration','campaign','pdays',
             'previous','poutcome','y']
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  FAIRAGENT - Retrain and Compare")
print("  Dataset: Bank Marketing")
print(f"  Protected: {PROTECTED_COL} (married=1 vs single=2)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load and encode full bank dataset
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading full bank dataset...")

try:
    # Try with header first
    df_raw = pd.read_csv(DATASET_PATH, sep=';')
    if LABEL_COL not in df_raw.columns:
        raise ValueError("No header")
    print("  Loaded with header")
except Exception:
    try:
        df_raw = pd.read_csv(
            DATASET_PATH, sep=';', names=BANK_COLS
        )
        print("  Loaded without header (UCI format)")
    except Exception:
        df_raw = pd.read_csv(DATASET_PATH)
        print("  Loaded with comma separator")

print(f"  Shape : {df_raw.shape}")
print(f"  Cols  : {list(df_raw.columns)}")

# Encode all categorical columns
encoders = {}
df_enc   = df_raw.copy()
for col in df_enc.columns:
    if df_enc[col].dtype == object or str(df_enc[col].dtype) == 'str':
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

print("  Encoded successfully")
if LABEL_COL in encoders:
    print(f"  y mapping    : {dict(zip(encoders[LABEL_COL].classes_, encoders[LABEL_COL].transform(encoders[LABEL_COL].classes_)))}")
if PROTECTED_COL in encoders:
    print(f"  marital map  : {dict(zip(encoders[PROTECTED_COL].classes_, encoders[PROTECTED_COL].transform(encoders[PROTECTED_COL].classes_)))}")

X_full = df_enc.drop(columns=[LABEL_COL])
y_full = df_enc[LABEL_COL]

print(f"  Full dataset : {X_full.shape[0]:,} rows")
print(f"  Label dist   : {y_full.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Split FIRST — test set LOCKED
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Splitting full dataset (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"  Train : {len(X_train):,} rows (80%)")
print(f"  Test  : {len(X_test):,} rows (20%) ← LOCKED forever")

# ─────────────────────────────────────────────────────────────
# STEP 3: Train ORIGINAL model on train split
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Training ORIGINAL model on train split...")

model_original = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=RANDOM_STATE
)
model_original.fit(X_train, y_train)

y_pred_original   = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"  Original accuracy : {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 4: SPD BEFORE retraining (on locked test set)
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Computing SPD BEFORE retraining...")

def calculate_spd(y_pred, X_data, col, priv_val, unpriv_val):
    """SPD = P(Y=1|privileged) - P(Y=1|unprivileged)"""
    priv_mask   = (X_data[col] == priv_val).values
    unpriv_mask = (X_data[col] == unpriv_val).values
    priv_pred   = y_pred[priv_mask]
    unpriv_pred = y_pred[unpriv_mask]
    if len(priv_pred) == 0 or len(unpriv_pred) == 0:
        print(f"  WARNING: one group empty for col='{col}'")
        return 0.0
    return round(float(priv_pred.mean() - unpriv_pred.mean()), 6)

spd_before = calculate_spd(
    y_pred_original, X_test,
    PROTECTED_COL, PRIV_VAL, UNPRIV_VAL
)
print(f"  SPD (marital) BEFORE : {spd_before:+.6f}")
print(f"  Privileged   : married (val={PRIV_VAL})")
print(f"  Unprivileged : single  (val={UNPRIV_VAL})")
print(f"  Bias level   : " + (
    "SEVERE"   if abs(spd_before) > 0.3 else
    "HIGH"     if abs(spd_before) > 0.2 else
    "MODERATE" if abs(spd_before) > 0.1 else "LOW"
))

# ─────────────────────────────────────────────────────────────
# STEP 5: Load discriminatory pairs from FairAgent
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Loading discriminatory pairs from FairAgent...")

try:
    disc_df = pd.read_csv(DISC_CSV)
except FileNotFoundError:
    print(f"  ERROR: '{DISC_CSV}' not found!")
    print("  Run FairAgent first.")
    exit(1)

print(f"  Loaded  : {disc_df.shape[0]:,} rows")
print(f"  Columns : {list(disc_df.columns)}")

# ExpGA CSV format: inp0_<feature>, label0, inp1_<feature>, label1
# Extract feature columns — all except label and sensitive_val columns
feature_cols = [c for c in X_train.columns]
inp0_cols = ["inp0_" + c for c in feature_cols]
inp1_cols = ["inp1_" + c for c in feature_cols]

# Check if ExpGA format (inp0_* columns present)
if inp0_cols[0] in disc_df.columns:
    X_inp0 = disc_df[inp0_cols].values
    X_inp1 = disc_df[inp1_cols].values
    # Both get same label — teaches model: marital should NOT affect prediction
    Y_inp0 = disc_df["label0"].values.astype(int)
    Y_inp1 = disc_df["label0"].values.astype(int)
    X_disc_arr = np.vstack([X_inp0, X_inp1])
    y_disc_arr = np.concatenate([Y_inp0, Y_inp1])
    X_disc = pd.DataFrame(X_disc_arr, columns=feature_cols)
    y_disc = pd.Series(y_disc_arr)
else:
    # Fallback: single-row format
    label_col_disc = LABEL_COL if LABEL_COL in disc_df.columns else 'y'
    X_disc = disc_df.drop(columns=[label_col_disc])
    y_disc = disc_df[label_col_disc].astype(int)
    meta_cols = ['sample_type','pair_id','patient_type','bias_type','explanation']
    for col in meta_cols:
        if col in X_disc.columns:
            X_disc = X_disc.drop(columns=[col])
    for col in X_train.columns:
        if col not in X_disc.columns:
            X_disc[col] = 0
    X_disc = X_disc[X_train.columns]

print(f"  Disc. pairs : {len(X_disc):,} rows")
print(f"  Label dist  : {y_disc.value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Combine train split + discriminatory pairs → retrain
# Test set NEVER touched
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Combining train + discriminatory pairs...")
print("      (test set LOCKED — never used in training)")

X_combined = pd.concat([X_train, X_disc], ignore_index=True)
y_combined  = pd.concat([y_train, y_disc], ignore_index=True)

print(f"  Bank train      : {len(X_train):,}")
print(f"  Disc. pairs     : {len(X_disc):,}")
print(f"  Combined total  : {len(X_combined):,}")
print(f"  Test (locked)   : {len(X_test):,}")

print("\n  Retraining MLP...")
model_retrained = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=RANDOM_STATE
)
model_retrained.fit(X_combined, y_combined)

y_pred_retrained   = model_retrained.predict(X_test)
accuracy_retrained = accuracy_score(y_test, y_pred_retrained)
print(f"  Retrained accuracy : {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)")

spd_after = calculate_spd(
    y_pred_retrained, X_test,
    PROTECTED_COL, PRIV_VAL, UNPRIV_VAL
)
print(f"  SPD (marital) AFTER  : {spd_after:+.6f}")

# ─────────────────────────────────────────────────────────────
# STEP 7: Comparison report + charts
# ─────────────────────────────────────────────────────────────
acc_change = accuracy_retrained - accuracy_original
spd_change = spd_after - spd_before

print("\n" + "=" * 60)
print("  COMPARISON REPORT")
print("  (both measured on same locked test set)")
print("=" * 60)
print(f"""
  ACCURACY:
  ─────────────────────────────────────────
  Before : {accuracy_original:.4f} ({accuracy_original*100:.2f}%)
  After  : {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)
  Change : {acc_change:+.4f} ({acc_change*100:+.2f}%)
  Result : {"Accuracy maintained ✓" if abs(acc_change) < 0.01
             else "Accuracy improved ✓" if acc_change > 0
             else "Accuracy decreased ✗"}

  SPD (marital - married vs single):
  ─────────────────────────────────────────
  Before : {spd_before:+.6f}
  After  : {spd_after:+.6f}
  Change : {spd_change:+.6f}
  Result : {"Bias REDUCED ✓" if abs(spd_after) < abs(spd_before)
             else "Bias INCREASED ✗" if abs(spd_after) > abs(spd_before)
             else "No change"}
""")

# Classification reports
print("  CLASSIFICATION REPORT - BEFORE:")
print("  " + "-" * 40)
for line in classification_report(
    y_test, y_pred_original,
    target_names=['No (0)', 'Yes (1)']
).split('\n'):
    print("  " + line)

print("  CLASSIFICATION REPORT - AFTER:")
print("  " + "-" * 40)
for line in classification_report(
    y_test, y_pred_retrained,
    target_names=['No (0)', 'Yes (1)']
).split('\n'):
    print("  " + line)

# ── Charts ─────────────────────────────────────────────────────
print("\n[7/7] Generating comparison charts...")

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Accuracy chart
ax1 = axes[0]
bars1 = ax1.bar(
    ['Before', 'After'],
    [accuracy_original * 100, accuracy_retrained * 100],
    color=['#3498DB', '#2ECC71'], width=0.4,
    alpha=0.85, edgecolor='black', linewidth=0.8
)
for bar, val in zip(bars1, [accuracy_original, accuracy_retrained]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.2,
             f'{val*100:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy Before vs After\nRetraining', fontweight='bold')
ax1.annotate(f'Change: {acc_change*100:+.2f}%',
             xy=(0.5, 0.05), xycoords='axes fraction',
             ha='center', fontsize=11,
             color='green' if acc_change >= 0 else 'red',
             fontweight='bold')

# SPD chart
ax2 = axes[1]
bars2 = ax2.bar(
    ['Before', 'After'],
    [spd_before, spd_after],
    color=['#E74C3C', '#F39C12'], width=0.4,
    alpha=0.85, edgecolor='black', linewidth=0.8
)
for bar, val in zip(bars2, [spd_before, spd_after]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 0.002,
             f'{val:+.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='green', linestyle='-',
            linewidth=2, label='Fair (SPD=0)')
ax2.set_ylabel('SPD (closer to 0 = fairer)')
ax2.set_title('SPD (marital)\nmarried vs single', fontweight='bold')
ax2.legend()
ax2.annotate(f'Change: {spd_change:+.6f}',
             xy=(0.5, 0.05), xycoords='axes fraction',
             ha='center', fontsize=11,
             color='green' if abs(spd_after) < abs(spd_before) else 'red',
             fontweight='bold')

fig.suptitle(
    'FairAgent: Retraining Impact\n'
    'Bank Marketing — protected: marital',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'retraining_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] {chart_path}")

# Save text report
report_path = os.path.join(OUTPUT_DIR, 'retraining_report.txt')
with open(report_path, 'w') as f:
    f.write("FAIRAGENT RETRAINING REPORT - Bank Marketing\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Full dataset       : {DATASET_PATH}\n")
    f.write(f"Disc. pairs CSV    : {DISC_CSV}\n")
    f.write(f"Protected attr     : {PROTECTED_COL}\n")
    f.write(f"Privileged value   : {PRIV_VAL} (married)\n")
    f.write(f"Unprivileged value : {UNPRIV_VAL} (single)\n")
    f.write(f"Bank train size    : {len(X_train):,}\n")
    f.write(f"Disc. pairs        : {len(X_disc):,}\n")
    f.write(f"Combined train     : {len(X_combined):,}\n")
    f.write(f"Test size (locked) : {len(X_test):,}\n\n")
    f.write(f"ACCURACY\n{'-'*40}\n")
    f.write(f"  Before : {accuracy_original:.4f}\n")
    f.write(f"  After  : {accuracy_retrained:.4f}\n")
    f.write(f"  Change : {acc_change:+.4f}\n\n")
    f.write(f"SPD (marital)\n{'-'*40}\n")
    f.write(f"  Before : {spd_before:+.6f}\n")
    f.write(f"  After  : {spd_after:+.6f}\n")
    f.write(f"  Change : {spd_change:+.6f}\n")

print(f"  [OK] {report_path}")
print("\n" + "=" * 60)
print("  [OK] Retraining analysis complete!")
print("=" * 60)