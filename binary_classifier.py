"""
=============================================================================
  WiFi CSI — Walking vs No Movement Classifier
  Binary Classification | Best Techniques | + Real-Time Prediction
=============================================================================

FOLDER STRUCTURE:
    dataset_raw/
        walking/        (15 txt files)
        no_movement/    (15 txt files)

RUN TRAINING:
    python binary_classifier.py --mode train

RUN REALTIME (reads live from a txt file being written):
    python binary_classifier.py --mode realtime --input live_csi.txt

RUN REALTIME ON A RECORDED FILE (simulate live):
    python binary_classifier.py --mode simulate --input walking_0006.txt
=============================================================================
"""

import os, glob, argparse, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive for training; switched later for realtime
import matplotlib.pyplot as plt
from collections import deque

from scipy.ndimage import median_filter
from scipy.signal import welch

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             RocCurveDisplay)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_ROOT     = "dataset_raw"
ACTIVITIES    = ["walking", "no_movement"]
N_SC          = 52
WINDOW_SIZE   = 80     # frames per window
WINDOW_STEP   = 20     # step (small = more windows = more data)
TOP_K_SC      = 20     # top subcarriers to keep

MODEL_PATH    = "models_binary/voting_clf.pkl"
SCALER_PATH   = "models_binary/scaler.pkl"
TOP_SC_PATH   = "models_binary/top_sc.npy"

COLORS = {"walking": "#2ecc71", "no_movement": "#3498db"}
os.makedirs("models_binary", exist_ok=True)
os.makedirs("plots_binary",  exist_ok=True)

# ─────────────────────────────────────────────────────────────────
#  1. LOAD
# ─────────────────────────────────────────────────────────────────
def load_files():
    records = []
    for act in ACTIVITIES:
        folder = os.path.join(DATA_ROOT, act)
        files  = sorted(glob.glob(os.path.join(folder, "*.txt")))
        print(f"  {act:15s}: {len(files):2d} files")
        for fp in files:
            try:
                df = pd.read_csv(fp, header=None)
                if df.shape[1] == N_SC:
                    records.append({"activity": act, "data": df.values.astype(float)})
            except: pass
    print(f"  → {len(records)} files total\n")
    return records

# ─────────────────────────────────────────────────────────────────
#  2. PREPROCESS
# ─────────────────────────────────────────────────────────────────
def preprocess_csi(data: np.ndarray) -> np.ndarray:
    c = data.copy()
    dead = np.std(c, axis=0) < 1e-6
    c[:, dead] = 0.0
    for i in range(c.shape[1]):
        col = c[:, i]
        if np.std(col) < 1e-6: continue
        mu, sig = np.mean(col), np.std(col)
        c[:, i] = np.clip(col, mu-3*sig, mu+3*sig)
        c[:, i] = median_filter(c[:, i], size=5)
    mn, mx = c.min(0), c.max(0)
    d = np.where((mx-mn) < 1e-6, 1.0, mx-mn)
    return (c - mn) / d

# ─────────────────────────────────────────────────────────────────
#  3. FEATURE EXTRACTION — Best for walking vs no_movement
#
#  Walking  → periodic oscillation, moderate speed, rhythmic FFT peak
#  No move  → nearly flat signal, near-zero diff, no periodicity
# ─────────────────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Per subcarrier (18 features):
      ENERGY    : mean, std, var, energy, range
      MOTION    : diff_mean, diff_var, diff_max (zero for no_move!)
      SHAPE     : skewness, kurtosis, zero_crossing_rate
      FREQUENCY : fft_peak_mag, fft_peak_freq, spectral_entropy
                  dominant_freq_power_ratio, band_low, band_mid
      PERIODICITY: autocorr_lag1 (high for walking, ~0 for no_move)
    GLOBAL (6):
      total_energy, global_diff_mean, global_diff_max,
      mean_subcarrier_activity, n_active_sc, cross_sc_correlation
    """
    T, C = window.shape
    feats = []
    diff  = np.diff(window, axis=0)

    for c in range(C):
        col  = window[:, c]
        dcol = diff[:, c]
        mu   = np.mean(col)
        sig  = np.std(col) + 1e-9

        # Energy
        feats.append(mu)
        feats.append(sig)
        feats.append(np.var(col))
        feats.append(np.sum(col**2))
        feats.append(np.max(col) - np.min(col))

        # Motion  ← KEY: no_movement ≈ 0 on all 3
        feats.append(np.mean(np.abs(dcol)))          # diff_mean
        feats.append(np.var(dcol))                   # diff_var
        feats.append(np.max(np.abs(dcol)))            # diff_max

        # Shape
        feats.append(np.mean(((col-mu)/sig)**3))     # skew
        feats.append(np.mean(((col-mu)/sig)**4)-3)   # kurtosis
        # Zero crossing rate — high for oscillating walking signal
        zcr = np.sum(np.diff(np.sign(col - mu)) != 0) / (T - 1)
        feats.append(zcr)

        # Frequency domain
        fft_mag = np.abs(np.fft.rfft(col))[1:]
        fft_freq = np.fft.rfftfreq(T)[1:]
        if len(fft_mag) > 3:
            n_f  = len(fft_mag)
            tot  = np.sum(fft_mag**2) + 1e-9
            pk   = np.argmax(fft_mag)
            feats.append(fft_mag[pk])                          # fft_peak_mag
            feats.append(fft_freq[pk])                         # fft_peak_freq
            # Spectral entropy — high = no periodicity (no_move = noise)
            p = fft_mag**2 / tot
            feats.append(-np.sum(p * np.log(p + 1e-12)))       # spectral_entropy
            feats.append(fft_mag[pk]**2 / tot)                 # dominant power ratio
            feats.append(np.sum(fft_mag[:n_f//3]**2)  / tot)   # low band
            feats.append(np.sum(fft_mag[n_f//3:2*n_f//3]**2) / tot)  # mid band
        else:
            feats.extend([0.0]*6)

        # Periodicity — autocorrelation at lag 1
        # Walking signal is periodic → high autocorr; no_move → near zero
        if sig > 1e-6:
            col_norm = (col - mu) / sig
            ac = np.correlate(col_norm, col_norm, mode="full")
            ac = ac[len(ac)//2:]
            ac /= (ac[0] + 1e-9)
            feats.append(ac[min(5, len(ac)-1)])               # autocorr_lag5
        else:
            feats.append(0.0)

    # ── GLOBAL features ──────────────────────────────────────────
    feats.append(np.sum(window**2))                            # total_energy
    feats.append(np.mean(np.abs(diff)))                        # global_diff_mean
    feats.append(np.max(np.abs(diff)))                         # global_diff_max

    per_sc_std = np.std(window, axis=0)
    feats.append(np.mean(per_sc_std))                          # mean SC activity
    feats.append(np.sum(per_sc_std > 0.05))                    # n_active_sc

    # Mean adjacent SC correlation
    active = np.where(per_sc_std > 1e-3)[0]
    if len(active) > 2:
        rs = [np.corrcoef(window[:,active[i]], window[:,active[i+1]])[0,1]
              for i in range(len(active)-1)]
        rs = [r for r in rs if not np.isnan(r)]
        feats.append(np.mean(rs) if rs else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats)

def build_feat_names(sc_indices):
    per_sc = ["mean","std","var","energy","range",
              "diff_mean","diff_var","diff_max",
              "skew","kurt","zcr",
              "fft_peak_mag","fft_peak_freq","spectral_entropy",
              "dom_power_ratio","band_low","band_mid",
              "autocorr_lag5"]
    names  = [f"SC{i}_{s}" for i in sc_indices for s in per_sc]
    names += ["g_energy","g_diff_mean","g_diff_max",
              "g_mean_activity","g_n_active_sc","g_adj_corr"]
    return names

def build_dataset(records, selected_sc=None):
    sc_idx = list(range(N_SC)) if selected_sc is None else list(selected_sc)
    X_list, y_list = [], []
    for rec in records:
        data = preprocess_csi(rec["data"])
        if selected_sc is not None:
            data = data[:, list(selected_sc)]
        n = data.shape[0]
        for s in range(0, n - WINDOW_SIZE + 1, WINDOW_STEP):
            X_list.append(extract_features(data[s:s+WINDOW_SIZE]))
            y_list.append(rec["activity"])
    return np.array(X_list), np.array(y_list), build_feat_names(sc_idx)

# ─────────────────────────────────────────────────────────────────
#  4. MODELS — Voting Ensemble (RF + GBM + SVM)
# ─────────────────────────────────────────────────────────────────
def build_models():
    rf  = RandomForestClassifier(n_estimators=400, max_depth=None,
                                  min_samples_split=3, class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              class_weight="balanced", probability=True, random_state=42)
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("svm", svm)],
        voting="soft", n_jobs=-1)
    return voting

# ─────────────────────────────────────────────────────────────────
#  5. PLOTS
# ─────────────────────────────────────────────────────────────────
def plot_scatter(X, y, title, fname, mode="pca"):
    Xs  = StandardScaler().fit_transform(X)
    if mode == "pca":
        pca = PCA(n_components=2, random_state=42)
        Z   = pca.fit_transform(Xs)
        xl  = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        yl  = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        Xs  = PCA(n_components=min(50,Xs.shape[1],Xs.shape[0]-1),
                  random_state=42).fit_transform(Xs)
        perp = min(30, max(5, len(Xs)//5))
        Z   = TSNE(n_components=2, perplexity=perp,
                   max_iter=1500, random_state=42).fit_transform(Xs)
        xl, yl = "t-SNE 1", "t-SNE 2"

    fig, ax = plt.subplots(figsize=(9, 7))
    for act in ACTIVITIES:
        mask = y == act
        ax.scatter(Z[mask,0], Z[mask,1], c=COLORS[act],
                   label=act.replace("_"," ").title(),
                   alpha=0.65, s=40, edgecolors="none")
        pts = Z[mask]
        if len(pts) >= 3:
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            rx, ry = pts[:,0].std()*2.2, pts[:,1].std()*2.2
            ax.add_patch(plt.matplotlib.patches.Ellipse(
                (cx,cy), rx*2, ry*2, fill=False,
                edgecolor=COLORS[act], lw=2, alpha=0.6, ls="--"))
            ax.annotate(act.replace("_","\n").title(), (cx,cy),
                        fontsize=10, ha="center", fontweight="bold",
                        color=COLORS[act],
                        bbox=dict(fc="white",alpha=0.7,ec=COLORS[act],pad=2))
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=12); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"plots_binary/{fname}", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots_binary/{fname}")

def plot_sc_importance(clf, feat_names, fname):
    # Extract RF importances from the voting ensemble
    rf_clf  = clf.estimators_[0]
    imp     = rf_clf.feature_importances_
    sc_imp  = np.zeros(N_SC)
    for fi, name in enumerate(feat_names[:len(imp)]):
        part = name.split("_")[0]
        if part.startswith("SC"):
            try:
                sc_imp[int(part[2:])] += imp[fi]
            except: pass
    sorted_idx = np.argsort(sc_imp)[::-1]
    fig, ax = plt.subplots(figsize=(16, 5))
    colors  = ["#e74c3c" if i < TOP_K_SC else "#95a5a6"
               for i in range(N_SC)]
    ax.bar(range(N_SC), sc_imp[sorted_idx],
           color=[colors[i] for i in range(N_SC)])
    ax.set_xticks(range(N_SC))
    ax.set_xticklabels([f"SC{i}" for i in sorted_idx], rotation=75, fontsize=7)
    ax.axvline(x=TOP_K_SC-0.5, color="black", lw=2, ls="--",
               label=f"Top-{TOP_K_SC} cutoff")
    ax.set_ylabel("Aggregated Importance", fontsize=11)
    ax.set_title("Subcarrier Importance — Walking vs No Movement",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots_binary/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots_binary/{fname}")
    return sorted_idx

def plot_confusion(clf, X_te, y_te, fname):
    y_pred = clf.predict(X_te)
    cm = confusion_matrix(y_te, y_pred, labels=ACTIVITIES)
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(cm, display_labels=ACTIVITIES).plot(
        ax=ax, cmap="Greens", colorbar=False)
    acc = (y_pred==y_te).mean()*100
    ax.set_title(f"Confusion Matrix — Acc: {acc:.1f}%",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots_binary/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots_binary/{fname}")

def plot_roc(clf, X_te, y_te, fname):
    y_score = clf.predict_proba(X_te)[:,ACTIVITIES.index("walking")]
    y_bin   = (y_te == "walking").astype(int)
    fig, ax = plt.subplots(figsize=(6,5))
    RocCurveDisplay.from_predictions(y_bin, y_score, ax=ax,
                                     color="#2ecc71", name="Walking vs No Movement")
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"plots_binary/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots_binary/{fname}")

def plot_feature_dist(records, fname):
    """Plot 4 key feature distributions per class — visual proof of separability."""
    feats_by_class = {a: [] for a in ACTIVITIES}
    for rec in records:
        data = preprocess_csi(rec["data"])
        n    = data.shape[0]
        for s in range(0, n - WINDOW_SIZE + 1, WINDOW_STEP):
            win  = data[s:s+WINDOW_SIZE]
            diff = np.diff(win, axis=0)
            feats_by_class[rec["activity"]].append({
                "Global Diff Mean\n(motion speed)":
                    np.mean(np.abs(diff)),
                "Mean ZCR\n(oscillation rate)":
                    np.mean([np.sum(np.diff(np.sign(win[:,c]-np.mean(win[:,c])))!=0)
                             /(WINDOW_SIZE-1) for c in range(N_SC)
                             if np.std(win[:,c])>1e-3]),
                "Mean Autocorr\n(periodicity)":
                    np.mean([np.corrcoef(win[:WINDOW_SIZE//2,c],
                                        win[WINDOW_SIZE//2:,c])[0,1]
                             for c in range(N_SC) if np.std(win[:,c])>1e-3]),
                "Spectral Entropy\n(frequency spread)":
                    np.mean([
                        -np.sum((lambda p: p)(
                            (lambda m: m**2/(np.sum(m**2)+1e-9))(
                                np.abs(np.fft.rfft(win[:,c]))[1:]))
                            * np.log(
                                (lambda m: m**2/(np.sum(m**2)+1e-9))(
                                    np.abs(np.fft.rfft(win[:,c]))[1:]) + 1e-12))
                        for c in range(N_SC) if np.std(win[:,c])>1e-3])
            })

    feat_keys = list(list(feats_by_class.values())[0][0].keys())
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, key in zip(axes, feat_keys):
        for act in ACTIVITIES:
            vals = [d[key] for d in feats_by_class[act]]
            ax.hist(vals, bins=25, alpha=0.6, color=COLORS[act],
                    label=act.replace("_"," ").title(), density=True)
        ax.set_title(key, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle("Key Feature Distributions — Walking vs No Movement",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots_binary/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: plots_binary/{fname}")

# ─────────────────────────────────────────────────────────────────
#  6. TRAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def train_pipeline():
    print("\n" + "="*60)
    print("  Binary Classifier: Walking vs No Movement")
    print("="*60)

    # Load
    print("\n[1] Loading files...")
    records = load_files()

    # Feature distribution plot
    print("[2] Plotting feature distributions...")
    plot_feature_dist(records, "00_feature_dist.png")

    # Build full dataset
    print("[3] Feature extraction (all 52 SC)...")
    X, y, feat_names = build_dataset(records)
    print(f"  Dataset: {X.shape} | {dict(zip(*np.unique(y, return_counts=True)))}")

    # Scatter before cleaning
    print("\n[4] Scatter plots — all SCs...")
    plot_scatter(X, y, "PCA — All 52 Subcarriers", "01_pca_all.png", mode="pca")
    plot_scatter(X, y, "t-SNE — All 52 Subcarriers", "02_tsne_all.png", mode="tsne")

    # Train initial RF for importance
    print("\n[5] Training RF for feature importance...")
    sc    = StandardScaler()
    X_s   = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2,
                                               stratify=y, random_state=42)
    rf_tmp = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=42)
    rf_tmp.fit(X_tr, y_tr)
    sorted_sc = plot_sc_importance(rf_tmp, feat_names, "03_sc_importance.png")
    top_sc    = sorted_sc[:TOP_K_SC]
    print(f"  Top-{TOP_K_SC} SCs: {list(top_sc)}")

    # Rebuild with top SCs
    print(f"\n[6] Rebuild dataset — Top-{TOP_K_SC} SCs only...")
    X_cl, y_cl, fn_cl = build_dataset(records, selected_sc=top_sc)
    print(f"  Cleaned: {X_cl.shape}")

    pd.DataFrame(X_cl, columns=fn_cl[:X_cl.shape[1]])\
      .assign(label=y_cl).to_csv("cleaned_binary.csv", index=False)

    # Scatter after cleaning
    print("\n[7] Scatter — cleaned...")
    plot_scatter(X_cl, y_cl,
                 f"PCA — Cleaned Top-{TOP_K_SC} SCs  ← Best Separability",
                 "04_pca_cleaned.png", mode="pca")
    plot_scatter(X_cl, y_cl,
                 f"t-SNE — Cleaned Top-{TOP_K_SC} SCs  ← Best Separability",
                 "05_tsne_cleaned.png", mode="tsne")

    # Train voting ensemble on cleaned
    print("\n[8] Training Voting Ensemble (RF + GBM + SVM)...")
    sc2   = StandardScaler()
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_cl, y_cl, test_size=0.2, stratify=y_cl, random_state=42)
    X_tr2_s = sc2.fit_transform(X_tr2)
    X_te2_s = sc2.transform(X_te2)

    clf = build_models()
    clf.fit(X_tr2_s, y_tr2)

    y_pred = clf.predict(X_te2_s)
    acc    = (y_pred == y_te2).mean() * 100
    print(f"\n  Ensemble Test Accuracy: {acc:.1f}%")
    print(classification_report(y_te2, y_pred, target_names=ACTIVITIES))

    # Cross validation
    sc_all = StandardScaler()
    X_cl_s = sc_all.fit_transform(X_cl)
    cv = cross_val_score(clf, X_cl_s, y_cl, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"  5-Fold CV: {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    plot_confusion(clf, X_te2_s, y_te2, "06_confusion.png")
    plot_roc(clf, X_te2_s, y_te2, "07_roc.png")

    # Save
    joblib.dump(clf,   MODEL_PATH)
    joblib.dump(sc2,   SCALER_PATH)
    np.save(TOP_SC_PATH, top_sc)
    print(f"\n  Models saved to models_binary/")
    print("="*60)
    print("  ✅ TRAINING DONE — run with --mode simulate to test realtime")
    print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────────
#  7. REAL-TIME PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
class RealtimePredictor:
    def __init__(self):
        print("\n  Loading model...")
        self.clf     = joblib.load(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)
        self.top_sc  = np.load(TOP_SC_PATH)
        self.buffer  = deque(maxlen=WINDOW_SIZE)
        self.history = []   # (timestamp, prediction, confidence)
        print(f"  Model loaded ✓  |  Using SCs: {list(self.top_sc[:8])}...")

    def ingest_row(self, row: np.ndarray) -> dict | None:
        """Feed one CSI row (52 values). Returns prediction when buffer is full."""
        self.buffer.append(row)
        if len(self.buffer) < WINDOW_SIZE:
            return None

        window = np.array(self.buffer)                    # (WINDOW_SIZE, N_SC)
        window = preprocess_csi(window)
        window = window[:, self.top_sc]                   # keep top SCs only
        feats  = extract_features(window).reshape(1, -1)
        feats_s= self.scaler.transform(feats)

        proba  = self.clf.predict_proba(feats_s)[0]
        pred   = ACTIVITIES[np.argmax(proba)]
        conf   = np.max(proba) * 100
        result = {"prediction": pred, "confidence": conf,
                  "proba": dict(zip(ACTIVITIES, proba*100))}
        self.history.append(result)
        return result

# ─────────────────────────────────────────────────────────────────
#  8. REAL-TIME DISPLAY (terminal + live plot)
# ─────────────────────────────────────────────────────────────────
def run_realtime(input_file: str, simulate: bool = False):
    """
    simulate=True  → read an existing file row by row (testing mode)
    simulate=False → tail a file being written live by your CSI hardware
    """
    if not os.path.exists(MODEL_PATH):
        print("  ❌ Model not found. Run --mode train first!")
        return

    predictor = RealtimePredictor()

    matplotlib.use("TkAgg") if not simulate else None

    # Live plot setup
    plt.ion()
    fig, (ax_sig, ax_bar) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("WiFi CSI — Real-Time Activity Monitor", fontsize=14, fontweight="bold")

    sig_lines = {}
    for act in ACTIVITIES:
        line, = ax_sig.plot([], [], label=act.replace("_"," ").title(),
                            color=COLORS[act], lw=1.5)
        sig_lines[act] = line

    conf_history = deque(maxlen=60)  # last 60 predictions
    time_axis    = list(range(60))

    bars = ax_bar.bar(ACTIVITIES, [0, 0],
                      color=[COLORS[a] for a in ACTIVITIES],
                      edgecolor="white", linewidth=2)

    ax_sig.set_xlim(0, WINDOW_SIZE)
    ax_sig.set_ylim(0, 1)
    ax_sig.set_title("CSI Signal — Top Subcarrier (SC0)", fontsize=11)
    ax_sig.set_xlabel("Frame"); ax_sig.set_ylabel("Normalized Amplitude")
    ax_sig.legend(fontsize=9)

    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Confidence (%)", fontsize=10)
    ax_bar.set_title("Prediction Confidence", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)

    pred_text = fig.text(0.5, 0.01, "", ha="center", fontsize=16,
                         fontweight="bold", color="black")

    signal_buf = deque(maxlen=WINDOW_SIZE)

    def update_plot(result, raw_row):
        signal_buf.append(raw_row[0])  # SC0 as representative signal

        # Update signal plot
        sig_lines[result["prediction"]].set_data(
            range(len(signal_buf)), list(signal_buf))
        ax_sig.set_xlim(0, WINDOW_SIZE)

        # Update confidence bars
        for bar, act in zip(bars, ACTIVITIES):
            bar.set_height(result["proba"][act])
            bar.set_color(COLORS[act] if act == result["prediction"] else "#bdc3c7")

        pred_text.set_text(
            f"🟢 {result['prediction'].replace('_',' ').upper()}  "
            f"({result['confidence']:.1f}% confidence)")
        pred_text.set_color(COLORS[result["prediction"]])

        fig.canvas.draw_idle()
        plt.pause(0.001)

    # ── READ MODE ────────────────────────────────────────────────
    print(f"\n  {'[SIMULATE]' if simulate else '[LIVE]'} Reading: {input_file}")
    print("  Press Ctrl+C to stop\n")
    print(f"  {'Frame':>6}  {'Prediction':>15}  {'Confidence':>12}  Probabilities")
    print("  " + "-"*60)

    if simulate:
        # Read file and replay row by row with a small delay
        df = pd.read_csv(input_file, header=None)
        rows = df.values.astype(float)
        for i, row in enumerate(rows):
            result = predictor.ingest_row(row)
            if result:
                proba_str = "  ".join([f"{a[:4]}:{p:.0f}%"
                                       for a, p in result["proba"].items()])
                flag = "✅" if result["confidence"] > 80 else "🟡"
                print(f"  {i:>6}  {result['prediction']:>15}  "
                      f"{result['confidence']:>10.1f}%  {proba_str}  {flag}")
                try:
                    update_plot(result, row)
                except Exception:
                    pass
            time.sleep(0.02)   # 50 fps simulation

    else:
        # Live mode: tail the file
        with open(input_file, "r") as f:
            f.seek(0, 2)   # seek to end
            frame = 0
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                try:
                    row = np.array([float(x) for x in line.strip().split(",")])
                    if len(row) != N_SC:
                        continue
                    result = predictor.ingest_row(row)
                    if result:
                        proba_str = "  ".join([f"{a[:4]}:{p:.0f}%"
                                               for a, p in result["proba"].items()])
                        print(f"  {frame:>6}  {result['prediction']:>15}  "
                              f"{result['confidence']:>10.1f}%  {proba_str}")
                        try:
                            update_plot(result, row)
                        except Exception:
                            pass
                    frame += 1
                except ValueError:
                    continue

    plt.ioff()
    print("\n  Done!")

    # ── SUMMARY PLOT ─────────────────────────────────────────────
    if predictor.history:
        preds   = [r["prediction"] for r in predictor.history]
        confs   = [r["confidence"] for r in predictor.history]
        from collections import Counter
        counts  = Counter(preds)
        majority = counts.most_common(1)[0]

        print(f"\n  ── SUMMARY ──────────────────────────")
        print(f"  Total predictions  : {len(preds)}")
        for act in ACTIVITIES:
            c = counts.get(act, 0)
            print(f"  {act:15s}: {c:4d}  ({c/len(preds)*100:.1f}%)")
        print(f"  Final verdict      : {majority[0].upper()}")
        print(f"  Mean confidence    : {np.mean(confs):.1f}%")
        print("  ─────────────────────────────────────\n")

        # Save summary plot
        plt.figure(figsize=(12, 4))
        colors = [COLORS[p] for p in preds]
        plt.bar(range(len(confs)), confs, color=colors, width=1.0, edgecolor="none")
        plt.axhline(80, color="black", ls="--", lw=1, label="80% threshold")
        for act in ACTIVITIES:
            plt.plot([], [], color=COLORS[act],
                     label=act.replace("_"," ").title(), linewidth=4)
        plt.xlabel("Prediction #"); plt.ylabel("Confidence (%)")
        plt.title("Real-Time Prediction History", fontweight="bold")
        plt.ylim(0, 105); plt.legend(); plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig("plots_binary/realtime_summary.png", dpi=150)
        plt.close()
        print("  Saved: plots_binary/realtime_summary.png")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","realtime","simulate"],
                        default="train", help="train | realtime | simulate")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file for realtime/simulate mode")
    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline()

    elif args.mode == "simulate":
        if not args.input:
            # Auto-pick first walking or no_movement file
            for folder in ["dataset_raw/walking", "dataset_raw/no_movement"]:
                files = glob.glob(os.path.join(folder, "*.txt"))
                if files:
                    args.input = files[0]
                    print(f"  Auto-selected: {args.input}")
                    break
        run_realtime(args.input, simulate=True)

    elif args.mode == "realtime":
        if not args.input:
            print("  ❌ Provide --input <live_csi_file.txt>")
        else:
            run_realtime(args.input, simulate=False)