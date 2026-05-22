"""
=============================================================================
  WiFi CSI — 3-Class Classifier
  Fall  |  Walking  |  No Movement
  Best features from binary (walking/no_move) + fall-specific temporal features
=============================================================================

FOLDER STRUCTURE:
    dataset_raw/
        fall/           (15 txt files)
        walking/        (15 txt files)
        no_movement/    (15 txt files)

RUN:
    python three_class_classifier.py
=============================================================================
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_ROOT   = "dataset_raw"
ACTIVITIES  = ["fall", "walking", "no_movement"]
N_SC        = 52
WINDOW_SIZE = 80
WINDOW_STEP = 20
TOP_K_SC    = 20

MODEL_DIR   = "models_3class"
PLOT_DIR    = "plots_3class"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

COLORS  = {"fall": "#e74c3c", "walking": "#2ecc71", "no_movement": "#3498db"}
MARKERS = {"fall": "o",       "walking": "s",        "no_movement": "^"}

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
                    records.append({"activity": act,
                                    "data": df.values.astype(float)})
            except Exception as e:
                print(f"    [ERR] {fp}: {e}")
    print(f"  → {len(records)} total files\n")
    return records

# ─────────────────────────────────────────────────────────────────
#  2. PREPROCESS
# ─────────────────────────────────────────────────────────────────
def preprocess_csi(data: np.ndarray) -> np.ndarray:
    c    = data.copy()
    dead = np.std(c, axis=0) < 1e-6
    c[:, dead] = 0.0
    for i in range(c.shape[1]):
        col = c[:, i]
        if np.std(col) < 1e-6: continue
        mu, sig = np.mean(col), np.std(col)
        c[:, i] = np.clip(col, mu - 3*sig, mu + 3*sig)
        c[:, i] = median_filter(c[:, i], size=5)
    mn, mx = c.min(0), c.max(0)
    return (c - mn) / np.where((mx - mn) < 1e-6, 1.0, mx - mn)

# ─────────────────────────────────────────────────────────────────
#  3. FEATURE EXTRACTION
#
#  Combined: binary features (great for walk/no_move) +
#            fall-specific temporal features (burst, impact, asymmetry)
# ─────────────────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Per subcarrier (24 features):
      STANDARD   : mean, std, var, energy, range
      MOTION     : diff_mean, diff_var, diff_max
      SHAPE      : skew, kurtosis, zcr
      FREQUENCY  : fft_peak_mag, fft_peak_freq, spectral_entropy,
                   dom_power_ratio, band_low, band_mid
      PERIODICITY: autocorr_lag5
      ★ FALL     : var_first_third, var_last_third, burst_ratio,
                   asymmetry, impact_score, rise_time

    GLOBAL (6):
      total_energy, global_diff_mean, global_diff_max,
      mean_sc_activity, n_active_sc, adj_corr
    """
    T, C  = window.shape
    feats = []
    diff  = np.diff(window, axis=0)

    h1 = T // 3        # end of first third
    h2 = T - T // 3    # start of last third

    for ci in range(C):
        col  = window[:, ci]
        dcol = diff[:, ci]
        mu   = np.mean(col)
        sig  = np.std(col) + 1e-9

        # Standard
        feats.append(mu)
        feats.append(sig)
        feats.append(np.var(col))
        feats.append(np.sum(col**2))
        feats.append(np.max(col) - np.min(col))

        # Motion
        feats.append(np.mean(np.abs(dcol)))
        feats.append(np.var(dcol))
        feats.append(np.max(np.abs(dcol)))

        # Shape
        feats.append(np.mean(((col - mu) / sig) ** 3))
        feats.append(np.mean(((col - mu) / sig) ** 4) - 3)
        feats.append(np.sum(np.diff(np.sign(col - mu)) != 0) / (T - 1))

        # Frequency
        fft_mag  = np.abs(np.fft.rfft(col))[1:]
        fft_freq = np.fft.rfftfreq(T)[1:]
        if len(fft_mag) > 3:
            n_f = len(fft_mag)
            tot = np.sum(fft_mag**2) + 1e-9
            pk  = np.argmax(fft_mag)
            p   = fft_mag**2 / tot
            feats.append(fft_mag[pk])
            feats.append(fft_freq[pk])
            feats.append(-np.sum(p * np.log(p + 1e-12)))
            feats.append(fft_mag[pk]**2 / tot)
            feats.append(np.sum(fft_mag[:n_f//3]**2) / tot)
            feats.append(np.sum(fft_mag[n_f//3:2*n_f//3]**2) / tot)
        else:
            feats.extend([0.0] * 6)

        # Periodicity
        if sig > 1e-6:
            cn = (col - mu) / sig
            ac = np.correlate(cn, cn, mode="full")[len(cn)-1:]
            ac /= (ac[0] + 1e-9)
            feats.append(ac[min(5, len(ac)-1)])
        else:
            feats.append(0.0)

        # ★ Fall-specific temporal features
        vf = np.var(col[:h1])   + 1e-9   # variance: first third
        vl = np.var(col[h2:])   + 1e-9   # variance: last third
        vm = np.var(col[h1:h2]) + 1e-9   # variance: middle third

        feats.append(vf)                          # var_first
        feats.append(vl)                          # var_last
        feats.append(vm / (vf + vl))             # burst_ratio  ★ fall=HIGH
        feats.append(abs(vf - vl) / (vf + vl))  # asymmetry    ★ fall=HIGH

        # Impact: max single frame change vs baseline noise
        bstd = np.std(dcol) + 1e-9
        feats.append(np.max(np.abs(dcol)) / bstd)  # impact_score ★

        # Rise time: how fast energy builds (fall=sudden, walk=gradual, no_move=slow)
        ce = np.cumsum((col - mu)**2)
        total_e = ce[-1] + 1e-9
        feats.append(np.searchsorted(ce / total_e, 0.5) / (T - 1))  # rise_time ★

    # Global features
    feats.append(np.sum(window**2))
    feats.append(np.mean(np.abs(diff)))
    feats.append(np.max(np.abs(diff)))
    psc = np.std(window, axis=0)
    feats.append(np.mean(psc))
    feats.append(float(np.sum(psc > 0.05)))
    active = np.where(psc > 1e-3)[0]
    if len(active) > 2:
        rs = [np.corrcoef(window[:, active[i]], window[:, active[i+1]])[0,1]
              for i in range(len(active)-1)]
        feats.append(np.mean([r for r in rs if not np.isnan(r)]) if rs else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats)

# ─────────────────────────────────────────────────────────────────
#  4. BUILD DATASET
# ─────────────────────────────────────────────────────────────────
def build_dataset(records, selected_sc=None):
    sc_idx = list(range(N_SC)) if selected_sc is None else list(selected_sc)
    X_list, y_list = [], []
    for rec in records:
        data = preprocess_csi(rec["data"])
        if selected_sc is not None:
            data = data[:, sc_idx]
        n = data.shape[0]
        for s in range(0, n - WINDOW_SIZE + 1, WINDOW_STEP):
            X_list.append(extract_features(data[s:s+WINDOW_SIZE]))
            y_list.append(rec["activity"])
    return np.array(X_list), np.array(y_list)

# ─────────────────────────────────────────────────────────────────
#  5. MODELS — Voting Ensemble
# ─────────────────────────────────────────────────────────────────
def build_ensemble():
    rf  = RandomForestClassifier(n_estimators=400, min_samples_split=3,
                                  class_weight="balanced",
                                  n_jobs=-1, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              class_weight="balanced", probability=True, random_state=42)
    return VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("svm", svm)],
        voting="soft", n_jobs=-1)

# ─────────────────────────────────────────────────────────────────
#  6. PLOTS
# ─────────────────────────────────────────────────────────────────
def plot_scatter(X, y, title, fname, mode="pca"):
    Xs = StandardScaler().fit_transform(X)
    if mode == "pca":
        pca = PCA(n_components=2, random_state=42)
        Z   = pca.fit_transform(Xs)
        xl  = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        yl  = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        Xs   = PCA(n_components=min(50, Xs.shape[1], Xs.shape[0]-1),
                   random_state=42).fit_transform(Xs)
        perp = min(30, max(5, len(Xs)//5))
        Z    = TSNE(n_components=2, perplexity=perp,
                    max_iter=1500, random_state=42).fit_transform(Xs)
        xl, yl = "t-SNE 1", "t-SNE 2"

    fig, ax = plt.subplots(figsize=(9, 7))
    for act in ACTIVITIES:
        mask = y == act
        ax.scatter(Z[mask,0], Z[mask,1],
                   c=COLORS[act], marker=MARKERS[act],
                   label=act.replace("_"," ").title(),
                   alpha=0.65, s=35, edgecolors="none")
        pts = Z[mask]
        if len(pts) >= 3:
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            rx, ry = pts[:,0].std()*2.2, pts[:,1].std()*2.2
            ax.add_patch(plt.matplotlib.patches.Ellipse(
                (cx,cy), rx*2, ry*2, fill=False,
                edgecolor=COLORS[act], lw=2, alpha=0.55, ls="--"))
            ax.annotate(act.replace("_","\n").title(), (cx, cy),
                        fontsize=9, ha="center", fontweight="bold",
                        color=COLORS[act],
                        bbox=dict(fc="white", alpha=0.7,
                                  ec=COLORS[act], pad=2))
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{fname}"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_sc_importance(clf, fname):
    rf_clf = clf.estimators_[0]
    imp    = rf_clf.feature_importances_
    # Each subcarrier has 24 features → aggregate per SC
    n_feats_per_sc = 24
    sc_imp = np.zeros(N_SC)
    for fi in range(min(len(imp), N_SC * n_feats_per_sc)):
        sc_imp[fi // n_feats_per_sc] += imp[fi]

    sorted_idx = np.argsort(sc_imp)[::-1]
    fig, ax = plt.subplots(figsize=(16, 5))
    cols = ["#e74c3c" if i < TOP_K_SC else "#bdc3c7" for i in range(N_SC)]
    ax.bar(range(N_SC), sc_imp[sorted_idx],
           color=[cols[i] for i in range(N_SC)])
    ax.set_xticks(range(N_SC))
    ax.set_xticklabels([f"SC{i}" for i in sorted_idx],
                       rotation=75, fontsize=7)
    ax.axvline(x=TOP_K_SC-0.5, color="black", lw=2, ls="--",
               label=f"Top-{TOP_K_SC} cutoff")
    ax.set_ylabel("Aggregated Importance", fontsize=11)
    ax.set_title("Subcarrier Importance — 3-Class",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOT_DIR}/{fname}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return sorted_idx

def plot_confusion(clf, X_te, y_te, fname):
    y_pred = clf.predict(X_te)
    cm     = confusion_matrix(y_te, y_pred, labels=ACTIVITIES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=ACTIVITIES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    acc = (y_pred == y_te).mean() * 100
    ax.set_title(f"Confusion Matrix — {acc:.1f}% Accuracy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOT_DIR}/{fname}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*58)
    print("  3-Class CSI Classifier: Fall | Walking | No Movement")
    print("="*58)

    # Load
    print("\n[1] Loading files...")
    records = load_files()
    if not records:
        print("No files found! Check dataset_raw/ folders.")
        return

    # Full feature extraction
    print("[2] Feature extraction (all 52 SCs, 24 feats/SC)...")
    X, y = build_dataset(records)
    print(f"  Dataset: {X.shape}")
    print(f"  Per class: { {a: int((y==a).sum()) for a in ACTIVITIES} }")

    # Scatter before cleaning
    print("\n[3] Scatter plots — all SCs...")
    plot_scatter(X, y, "PCA — All 52 Subcarriers (3-Class)", "01_pca_all.png", "pca")
    plot_scatter(X, y, "t-SNE — All 52 Subcarriers (3-Class)", "02_tsne_all.png", "tsne")

    # Quick RF for importance
    print("\n[4] Feature importance...")
    sc_tmp  = StandardScaler()
    X_s     = sc_tmp.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.2, stratify=y, random_state=42)
    rf_tmp  = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=42)
    rf_tmp.fit(X_tr, y_tr)
    sorted_sc = plot_sc_importance(
        type("_", (), {"estimators_": [rf_tmp]})(), "03_sc_importance.png")
    top_sc = sorted_sc[:TOP_K_SC]
    print(f"  Top-{TOP_K_SC} SCs: {list(top_sc)}")

    # Rebuild with top SCs
    print(f"\n[5] Rebuild dataset — Top-{TOP_K_SC} SCs...")
    X_cl, y_cl = build_dataset(records, selected_sc=top_sc)
    print(f"  Cleaned: {X_cl.shape}")

    # Scatter after cleaning
    print("\n[6] Scatter — cleaned...")
    plot_scatter(X_cl, y_cl,
                 f"PCA — Cleaned Top-{TOP_K_SC} SCs  ★ Show Sir",
                 "04_pca_cleaned.png", "pca")
    plot_scatter(X_cl, y_cl,
                 f"t-SNE — Cleaned Top-{TOP_K_SC} SCs  ★ Show Sir",
                 "05_tsne_cleaned.png", "tsne")

    # Train ensemble
    print("\n[7] Training Voting Ensemble (RF + GBM + SVM)...")
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_cl, y_cl, test_size=0.2, stratify=y_cl, random_state=42)
    sc2      = StandardScaler()
    X_tr2_s  = sc2.fit_transform(X_tr2)
    X_te2_s  = sc2.transform(X_te2)

    clf = build_ensemble()
    clf.fit(X_tr2_s, y_tr2)

    y_pred = clf.predict(X_te2_s)
    acc    = (y_pred == y_te2).mean() * 100
    print(f"\n  Test Accuracy : {acc:.1f}%")
    print(classification_report(y_te2, y_pred, target_names=ACTIVITIES))

    # Cross validation
    sc_all  = StandardScaler()
    X_cl_s  = sc_all.fit_transform(X_cl)
    cv = cross_val_score(clf, X_cl_s, y_cl,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"  5-Fold CV     : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    plot_confusion(clf, X_te2_s, y_te2, "06_confusion.png")

    # Save
    joblib.dump(clf,    f"{MODEL_DIR}/voting_clf.pkl")
    joblib.dump(sc2,    f"{MODEL_DIR}/scaler.pkl")
    np.save(f"{MODEL_DIR}/top_sc.npy", top_sc)
    print(f"\n  Models saved to {MODEL_DIR}/")

    # Print classes order (important for realtime!)
    print(f"  clf.classes_ order: {list(clf.classes_)}")

    print("\n" + "="*58)
    print("  DONE! Run realtime_3class.py to predict live")
    print("="*58 + "\n")

if __name__ == "__main__":
    main()