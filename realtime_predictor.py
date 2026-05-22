"""
=============================================================================
  WiFi CSI — Real-Time Activity Predictor  (ESP32)
  Predicts every 1 second | Majority vote over last 3 → stable output
=============================================================================
  RUN:
      python realtime_predictor.py
      python realtime_predictor.py --port COM3
=============================================================================
"""

import serial
import serial.tools.list_ports
import numpy as np
import argparse, time, os, sys
from collections import deque, Counter
from scipy.ndimage import median_filter
import joblib

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
BAUD_RATE      = 115200
N_SC           = 52
ACTIVITIES     = ["walking", "no_movement"]
WINDOW_SIZE    = 80     # must match training
PREDICT_EVERY  = 43     # ~1 second at 43 fps
VOTE_WINDOW    = 3      # majority vote over last N predictions → stability

MODEL_PATH  = "models_binary/voting_clf.pkl"
SCALER_PATH = "models_binary/scaler.pkl"
TOP_SC_PATH = "models_binary/top_sc.npy"

# ─────────────────────────────────────────────────────────────────
#  PREPROCESSING + FEATURES  (must match binary_classifier.py)
# ─────────────────────────────────────────────────────────────────
def preprocess_csi(data):
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

def extract_features(window):
    T, NC = window.shape
    feats = []
    diff  = np.diff(window, axis=0)

    for ci in range(NC):
        col  = window[:, ci]
        dcol = diff[:, ci]
        mu   = np.mean(col)
        sig  = np.std(col) + 1e-9

        feats += [mu, sig, np.var(col), np.sum(col**2),
                  np.max(col) - np.min(col),
                  np.mean(np.abs(dcol)), np.var(dcol), np.max(np.abs(dcol)),
                  np.mean(((col-mu)/sig)**3),
                  np.mean(((col-mu)/sig)**4) - 3,
                  np.sum(np.diff(np.sign(col-mu)) != 0) / (T-1)]

        fft_mag  = np.abs(np.fft.rfft(col))[1:]
        fft_freq = np.fft.rfftfreq(T)[1:]
        if len(fft_mag) > 3:
            n_f = len(fft_mag)
            tot = np.sum(fft_mag**2) + 1e-9
            pk  = np.argmax(fft_mag)
            p   = fft_mag**2 / tot
            feats += [fft_mag[pk], fft_freq[pk],
                      -np.sum(p * np.log(p + 1e-12)),
                      fft_mag[pk]**2 / tot,
                      np.sum(fft_mag[:n_f//3]**2) / tot,
                      np.sum(fft_mag[n_f//3:2*n_f//3]**2) / tot]
        else:
            feats += [0.0] * 6

        if sig > 1e-6:
            cn = (col - mu) / sig
            ac = np.correlate(cn, cn, mode="full")[len(cn)-1:]
            ac /= (ac[0] + 1e-9)
            feats.append(ac[min(5, len(ac)-1)])
        else:
            feats.append(0.0)

    feats += [np.sum(window**2), np.mean(np.abs(diff)), np.max(np.abs(diff))]
    psc = np.std(window, axis=0)
    feats += [np.mean(psc), float(np.sum(psc > 0.05))]
    active = np.where(psc > 1e-3)[0]
    if len(active) > 2:
        rs = [np.corrcoef(window[:, active[i]], window[:, active[i+1]])[0,1]
              for i in range(len(active)-1)]
        feats.append(np.mean([r for r in rs if not np.isnan(r)]) if rs else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats)

# ─────────────────────────────────────────────────────────────────
#  AUTO-DETECT PORT
# ─────────────────────────────────────────────────────────────────
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    print(f"  Found {len(ports)} port(s):")
    for p in ports:
        print(f"    {p.device} — {p.description}")
    for p in ports:
        if any(k in p.description.lower()
               for k in ["cp210", "ch340", "ch341", "uart", "esp32"]):
            print(f"  Auto-detected: {p.device}\n")
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None

# ─────────────────────────────────────────────────────────────────
#  DISPLAY — just the activity, clean and simple
# ─────────────────────────────────────────────────────────────────
EMOJIS = {"walking": "🚶", "no_movement": "🧍"}

def show(pred, conf, pred_count, total_frames, voted):
    # ANSI: clear current line + 4 lines above (overwrite previous output)
    if pred_count > 1:
        print("\033[5A\033[J", end="")

    label = pred.replace("_", " ").upper()
    emoji = EMOJIS[pred]

    # Color: green = walking, blue = no_movement
    color = "\033[92m" if pred == "walking" else "\033[94m"
    reset = "\033[0m"
    bold  = "\033[1m"
    dim   = "\033[2m"

    print(f"  {'─'*38}")
    print(f"  {bold}#{pred_count:03d}{reset}  "
          f"{emoji}  {bold}{color}{label}{reset}"
          f"  {dim}({conf:.0f}%){reset}")
    print(f"  {dim}frames: {total_frames}   "
          f"{'[voted]' if voted else '[single]'}{reset}")
    print(f"  {'─'*38}")

def show_collecting(n, total):
    pct    = n / PREDICT_EVERY
    filled = int(pct * 20)
    bar    = "▓" * filled + "░" * (20 - filled)
    print(f"\r  [{bar}] {n:2d}/{PREDICT_EVERY} frames",
          end="", flush=True)

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def run(port):
    print(f"\n  Loading model...")
    for p in [MODEL_PATH, SCALER_PATH, TOP_SC_PATH]:
        if not os.path.exists(p):
            print(f"  Not found: {p}\n  Train first: python binary_classifier.py --mode train")
            sys.exit(1)

    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    top_sc = np.load(TOP_SC_PATH)
    print(f"  Model loaded ✓")

    print(f"  Connecting to {port}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"  Cannot open port: {e}")
        print(f"  → Close Arduino Serial Monitor first!")
        sys.exit(1)
    print(f"  Connected ✓\n")

    ser.write(b"START\n")
    time.sleep(0.3)

    print(f"  ════════════════════════════════════════")
    print(f"  \033[1mWiFi CSI — Live Activity Detection\033[0m")
    print(f"  Predicts every ~1 sec | Majority vote: {VOTE_WINDOW}")
    print(f"  Ctrl+C to stop")
    print(f"  ════════════════════════════════════════\n")

    rolling       = deque(maxlen=WINDOW_SIZE)
    recent_preds  = deque(maxlen=VOTE_WINDOW)   # last N raw predictions
    frames_since  = 0
    total_frames  = 0
    pred_count    = 0
    history       = []

    try:
        while True:
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not raw.startswith("CSI_DATA,"):
                continue

            try:
                parts = raw[len("CSI_DATA,"):].split(",")
                if len(parts) != N_SC: continue
                row = np.array([float(v) for v in parts])
            except ValueError:
                continue

            rolling.append(row)
            total_frames  += 1
            frames_since  += 1

            show_collecting(frames_since, total_frames)

            # ── Predict every ~1 second ───────────────────────────
            if frames_since >= PREDICT_EVERY and len(rolling) == WINDOW_SIZE:
                frames_since = 0
                print()   # end the collecting bar line

                window  = preprocess_csi(np.array(rolling))[:, top_sc]
                feats_s = scaler.transform(extract_features(window).reshape(1,-1))
                proba   = clf.predict_proba(feats_s)[0]
                raw_pred = clf.classes_[np.argmax(proba)]  # fix: sklearn sorts classes alphabetically
                conf     = np.max(proba) * 100

                # ── Majority vote over last VOTE_WINDOW predictions ──
                recent_preds.append(raw_pred)
                voted_pred = Counter(recent_preds).most_common(1)[0][0]
                voted      = len(recent_preds) == VOTE_WINDOW

                pred_count += 1
                history.append(voted_pred)

                show(voted_pred, conf, pred_count, total_frames, voted)

    except KeyboardInterrupt:
        print(f"\n\n  Stopping...")
    finally:
        try:
            ser.write(b"STOP\n")
            time.sleep(0.2)
            ser.close()
        except Exception:
            pass

        if history:
            counts  = Counter(history)
            verdict = counts.most_common(1)[0][0]
            print(f"\n  ── Summary {'─'*28}")
            print(f"  Total predictions : {len(history)}")
            for act in ACTIVITIES:
                n = counts.get(act, 0)
                print(f"  {act:15s}: {n}x  "
                      f"({n/len(history)*100:.0f}%)")
            print(f"\n  Verdict → \033[1m{verdict.replace('_',' ').upper()}\033[0m")
            print(f"  {'─'*38}\n")

# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None)
    args = parser.parse_args()

    print("  ════════════════════════════════════════")
    print("  \033[1mWiFi CSI Real-Time Predictor\033[0m")
    print("  ════════════════════════════════════════")

    port = args.port or find_esp32_port()
    if not port:
        print("\n  No port found. Run: python realtime_predictor.py --port COM3")
        sys.exit(1)

    run(port)