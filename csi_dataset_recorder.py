"""
╔══════════════════════════════════════════════════════════╗
║         WiFi CSI DATASET COLLECTOR                      ║
║         Matched to your ESP32 code exactly              ║
╠══════════════════════════════════════════════════════════╣
║                                                         ║
║  ✏️  EDIT THE SETTINGS SECTION BELOW BEFORE RUNNING!   ║
║                                                         ║
║  Flow:                                                  ║
║    Press Enter → 3s countdown → CAPTURE X seconds       ║
║    → See waveform → Keep [K] or Delete [D] → Repeat    ║
║                                                         ║
╚══════════════════════════════════════════════════════════╝
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

# ╔══════════════════════════════════════════════════════════╗
# ║   ✏️  EDIT HERE  ─  change before every session        ║
# ╚══════════════════════════════════════════════════════════╝

PORT         = "COM5"           # ← your COM port
BAUDRATE     = 115200

# Activity: "fall" | "walking" | "no_movement"
ACTIVITY     = "walking"   # ← change for each activity you want to capture

# How many seconds to record per sample
DURATION_SEC = 10

# Output folder  →  will save to  OUTPUT_DIR / ACTIVITY /
OUTPUT_DIR   = "dataset_raw"

# ╔══════════════════════════════════════════════════════════╗
# ║  Don't edit below unless you know what you're doing     ║
# ╚══════════════════════════════════════════════════════════╝

NUM_SUBCARRIERS = 52   # matches your ESP32 code (52 subcarriers)

ACTIVITY_COLOR = {
    "fall":        "#FF4444",
    "walking":     "#44DD44",
    "no_movement": "#4499FF",
}

ACTIVITY_HINT = {
    "fall":
        "✅ GOOD: flat start → BIG spike/drop in middle → flat again",
    "walking":
        "✅ GOOD: continuous up/down oscillations the whole time",
    "no_movement":
        "✅ GOOD: nearly FLAT from start to end (very low variance)",
}

# Quality thresholds for automatic analysis
QUALITY_THRESHOLDS = {
    "fall": {
        "std_min": 10.0,
        "std_max": 50.0,
        "variance_min": 100.0,
    },
    "walking": {
        "std_min": 5.0,
        "std_max": 25.0,
        "variance_min": 25.0,
    },
    "no_movement": {
        "std_min": 0.0,
        "std_max": 8.0,
        "variance_max": 64.0,
    },
}

COLOR    = ACTIVITY_COLOR.get(ACTIVITY, "#FFFFFF")
HINT     = ACTIVITY_HINT.get(ACTIVITY, "")
SAVE_DIR = os.path.join(OUTPUT_DIR, ACTIVITY)
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
def analyze_quality(data: np.ndarray, activity: str) -> tuple:
    """
    Automatically analyze if the captured data is GOOD or BAD
    Returns: (is_good: bool, message: str, details: str)
    """
    mean_sig = data.mean(axis=1)
    overall_std = data.std()
    overall_var = data.var()
    mean_std = mean_sig.std()
    
    thresholds = QUALITY_THRESHOLDS.get(activity, {})
    issues = []
    
    if activity == "no_movement":
        # Should be FLAT (low std)
        if overall_std > thresholds.get("std_max", 8.0):
            issues.append(f"Too much movement (std={overall_std:.1f}, should be <8)")
        if overall_var > thresholds.get("variance_max", 64.0):
            issues.append(f"High variance (var={overall_var:.1f}, should be <64)")
        
        # Check if actually flat
        changes = np.abs(np.diff(mean_sig))
        if np.max(changes) > 10.0:
            issues.append("Large spikes detected - person was moving!")
    
    elif activity == "walking":
        # Should have RHYTHMIC oscillations (medium std)
        if overall_std < thresholds.get("std_min", 5.0):
            issues.append(f"Too flat (std={overall_std:.1f}, should be >5)")
        elif overall_std > thresholds.get("std_max", 25.0):
            issues.append(f"Too noisy (std={overall_std:.1f}, should be <25)")
        
        if overall_var < thresholds.get("variance_min", 25.0):
            issues.append("Not enough movement - walk more actively!")
        
        # Check for oscillations
        changes = np.abs(np.diff(mean_sig))
        if np.std(changes) < 2.0:
            issues.append("No rhythmic pattern - keep walking throughout!")
    
    elif activity == "fall":
        # Should have BIG spike (high std)
        if overall_std < thresholds.get("std_min", 10.0):
            issues.append(f"No fall detected (std={overall_std:.1f}, should be >10)")
        
        if overall_var < thresholds.get("variance_min", 100.0):
            issues.append("Variance too low - fall motion not captured")
        
        # Check for spike
        changes = np.abs(np.diff(mean_sig))
        max_change = np.max(changes)
        if max_change < 15.0:
            issues.append(f"No sudden spike (max change={max_change:.1f}, should be >15)")
    
    # Verdict
    is_good = len(issues) == 0
    
    if is_good:
        message = "✅ GOOD QUALITY!"
        details = (f"Std: {overall_std:.2f}  |  Var: {overall_var:.2f}  |  "
                  f"Perfect for {activity}!")
    else:
        message = "❌ BAD QUALITY"
        details = "\n      ".join(issues)
    
    return is_good, message, details

# ─────────────────────────────────────────────────────────
def count_saved():
    return len([f for f in os.listdir(SAVE_DIR) if f.endswith(".txt")])

# ─────────────────────────────────────────────────────────
def parse_line(line: str):
    """
    Your ESP32 sends:  CSI_DATA,12.3,45.6,...   (20 floats, comma-separated)
    Returns numpy array of 20 values, or None if invalid.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    parts = line.split(",")
    # parts[0] = "CSI_DATA", parts[1..20] = float values
    if len(parts) < NUM_SUBCARRIERS + 1:
        return None
    try:
        vals = [float(p) for p in parts[1 : NUM_SUBCARRIERS + 1]]
        return np.array(vals, dtype=np.float32)
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────
def show_waveform(data: np.ndarray, sample_idx: int, is_good: bool = None) -> None:
    """
    Show 3-panel waveform plot immediately after capture.
    data shape: (frames, subcarriers)
    is_good: optional quality indicator to display
    """
    fig = plt.figure(figsize=(14, 7), facecolor="#0d1117")
    
    # Add quality indicator to window title if provided
    quality_str = ""
    if is_good is not None:
        quality_str = "  ✅ GOOD" if is_good else "  ❌ BAD"
    
    fig.canvas.manager.set_window_title(
        f"Sample #{sample_idx:04d}  |  {ACTIVITY.upper()}{quality_str}  "
        f"|  Keep [K] or Delete [D]?")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── Panel 1: all subcarriers ───────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#16213e")
    cmap = plt.cm.viridis(np.linspace(0, 1, NUM_SUBCARRIERS))
    for i in range(NUM_SUBCARRIERS):
        ax1.plot(data[:, i], color=cmap[i], alpha=0.55, linewidth=0.9)
    ax1.set_title(
        f"All {NUM_SUBCARRIERS} subcarriers   |   "
        f"Frames: {len(data)}   |   "
        f"Std: {data.std():.2f}",
        color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Frame index", color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("Amplitude", color="#aaaaaa", fontsize=9)
    ax1.tick_params(colors="white", labelsize=8)
    for sp in ax1.spines.values():
        sp.set_color("#444444")

    # ── Panel 2: mean signal ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#16213e")
    mean_sig = data.mean(axis=1)
    ax2.plot(mean_sig, color=COLOR, linewidth=1.6)
    ax2.axhline(mean_sig.mean(), color="yellow",
                linestyle="--", alpha=0.45, linewidth=1)
    ax2.set_title(
        f"Mean signal\nmean={mean_sig.mean():.1f}  std={mean_sig.std():.2f}",
        color="white", fontsize=10, pad=6)
    ax2.set_xlabel("Frame index", color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Amplitude", color="#aaaaaa", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#444444")

    # ── Panel 3: per-frame variance ───────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#16213e")
    var_sig = data.var(axis=1)
    ax3.plot(var_sig, color="#FFAA00", linewidth=1.6)
    ax3.axhline(var_sig.mean(), color="cyan",
                linestyle="--", alpha=0.45, linewidth=1)
    ax3.set_title(
        f"Per-frame variance\navg={var_sig.mean():.2f}  max={var_sig.max():.2f}",
        color="white", fontsize=10, pad=6)
    ax3.set_xlabel("Frame index", color="#aaaaaa", fontsize=9)
    ax3.set_ylabel("Variance", color="#aaaaaa", fontsize=9)
    ax3.tick_params(colors="white", labelsize=8)
    for sp in ax3.spines.values():
        sp.set_color("#444444")

    # ── Title bar ─────────────────────────────────────────
    fig.text(0.5, 0.97,
             f"{ACTIVITY.upper()}  —  Sample #{sample_idx:04d}  "
             f"—  {DURATION_SEC}s recording",
             ha="center", va="top",
             color=COLOR, fontsize=13, fontweight="bold")

    # ── Hint at bottom ────────────────────────────────────
    fig.text(0.5, 0.01, HINT,
             ha="center", va="bottom",
             color="#999999", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.5)

# ─────────────────────────────────────────────────────────
# CONNECT TO ESP32
# ─────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════╗")
print("║        WiFi CSI  DATASET  COLLECTOR             ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  Activity  :  {ACTIVITY.upper():<35s}║")
print(f"║  Duration  :  {str(DURATION_SEC)+' seconds':<35s}║")
print(f"║  Save to   :  {SAVE_DIR:<35s}║")
print(f"║  COM Port  :  {PORT:<35s}║")
print("╚══════════════════════════════════════════════════╝")
print()

print("📡 Connecting to ESP32 ...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"✅ Connected on {PORT}")
    print()
except Exception as e:
    print(f"❌ Cannot open {PORT}:  {e}")
    print("   → Close Arduino Serial Monitor first!")
    exit(1)

# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────
sample_idx = count_saved() + 1

try:
    while True:

        # ── Prompt ───────────────────────────────────────
        print("─" * 52)
        print(f"  Sample    : #{sample_idx:04d}")
        print(f"  Activity  : {ACTIVITY.upper()}")
        print(f"  Saved     : {count_saved()} samples so far in {SAVE_DIR}/")
        print("─" * 52)
        print("  Press ENTER to start   |   Q + ENTER to quit")
        print()

        user = input("  → ").strip().lower()
        if user == "q":
            print("\n👋 Quitting...\n")
            break

        # ── 3-second countdown ───────────────────────────
        print()
        for n in range(3, 0, -1):
            print(f"  ⏳  Get ready ...  {n}  ", end="\r", flush=True)
            time.sleep(1)
        print(f"  🔴  RECORDING — Perform {ACTIVITY.upper()} NOW!          ")
        print()

        # ── Send START command to your ESP32 ─────────────
        # Your ESP32 starts CSI + traffic on receiving "START"
        ser.reset_input_buffer()
        ser.write(b"START\n")

        # Let ESP32 initialise, flush the "CSI_STARTED" line
        time.sleep(0.3)
        while ser.in_waiting:
            ser.readline()

        # ── Capture loop ─────────────────────────────────
        frames  = []
        t_start = time.perf_counter()

        while True:
            elapsed   = time.perf_counter() - t_start
            remaining = max(0.0, DURATION_SEC - elapsed)

            if elapsed >= DURATION_SEC:
                break

            # Progress bar
            filled = int((elapsed / DURATION_SEC) * 46)
            bar    = "█" * filled + "░" * (46 - filled)
            print(f"  [{bar}]  {remaining:.1f}s  ({len(frames)} frames)",
                  end="\r", flush=True)

            try:
                raw = ser.readline().decode(errors="ignore")
            except Exception:
                continue

            row = parse_line(raw)
            if row is not None:
                frames.append(row)

        # ── Send STOP to ESP32 ────────────────────────────
        # Your ESP32 stops CSI + traffic on receiving "STOP"
        ser.write(b"STOP\n")
        time.sleep(0.2)
        ser.reset_input_buffer()

        n_frames = len(frames)
        print(f"\n\n  ✅  Captured {n_frames} frames  "
              f"({n_frames / DURATION_SEC:.0f} fps)")

        if n_frames < 5:
            print("  ⚠️  Too few frames — check ESP32 and retry\n")
            continue

        data = np.array(frames)     # shape: (n_frames, 20)

        # ── Save file ─────────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ACTIVITY}_{sample_idx:04d}_{ts}.txt"
        filepath = os.path.join(SAVE_DIR, filename)
        np.savetxt(filepath, data, delimiter=",", fmt="%.1f")

        # ── Automatic Quality Analysis ────────────────────
        is_good, quality_msg, quality_details = analyze_quality(data, ACTIVITY)
        
        # ── Show waveform with quality indicator ──────────
        print()
        print("  📊 Analyzing waveform...")
        print()
        show_waveform(data, sample_idx, is_good)
        
        print()
        print("  " + "═" * 50)
        print(f"  🤖 AUTOMATIC QUALITY ANALYSIS")
        print("  " + "═" * 50)
        
        if is_good:
            print(f"  {quality_msg}")
            print(f"  {quality_details}")
            print(f"  → Recommendation: KEEP this sample! ✅")
        else:
            print(f"  {quality_msg}")
            print(f"  Issues found:")
            print(f"      {quality_details}")
            print(f"  → Recommendation: DELETE and try again! ❌")
        
        print("  " + "═" * 50)
        print()

        # ── Keep or Delete ────────────────────────────────
        print(f"  File: {filename}")
        print(f"  Frames: {n_frames}  |  Std: {data.std():.2f}  "
              f"|  Mean: {data.mean():.2f}")
        print()

        while True:
            choice = input("  KEEP [K]  or  DELETE [D] ?   → ").strip().lower()

            if choice == "k":
                print(f"\n  ✅  KEPT  →  {filepath}\n")
                sample_idx += 1
                plt.close("all")
                break

            elif choice == "d":
                os.remove(filepath)
                print(f"\n  ❌  DELETED  —  try again\n")
                plt.close("all")
                break

            else:
                print("  Type  K  to keep  or  D  to delete")

        time.sleep(0.3)

# ─────────────────────────────────────────────────────────
except KeyboardInterrupt:
    print("\n\n  ⛔  Stopped")

finally:
    try:
        ser.write(b"STOP\n")
        ser.close()
    except Exception:
        pass

    saved = count_saved()
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║              SESSION COMPLETE                   ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Activity  :  {ACTIVITY.upper():<35s}║")
    print(f"║  Samples   :  {str(saved)+' files saved':<35s}║")
    print(f"║  Folder    :  {SAVE_DIR:<35s}║")
    print("╚══════════════════════════════════════════════════╝")
    print() 