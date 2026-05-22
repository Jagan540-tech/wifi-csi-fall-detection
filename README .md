# Human Activity Recognition Through WiFi CSI Signals

> Real-time fall detection, walking, and no-movement classification using WiFi Channel State Information — no cameras, no wearables.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32-Microcontroller-E7352C?style=flat&logo=espressif&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Voting%20Ensemble-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.1%25-00C853?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

---

## Overview

This project presents a non-invasive Human Activity Recognition (HAR) system that detects human activities in real time using WiFi Channel State Information (CSI) signals captured by an ESP32 microcontroller. The system classifies three activities — Fall, Walking, and No Movement — without requiring cameras or wearable sensors, preserving complete user privacy.

When a fall is detected, an automated Telegram emergency alert is instantly sent to designated caregivers, making it suitable for elderly care and smart home monitoring applications.

---

## Features

- 52-subcarrier CSI capture at approximately 43 frames per second via ESP32
- 4-stage preprocessing pipeline — dead subcarrier removal, outlier clipping, median filtering, and normalization
- 24 handcrafted features per subcarrier including fall-specific temporal descriptors
- Soft Voting Ensemble of Random Forest, Gradient Boosting, and Support Vector Machine classifiers
- Dual detection pipeline — sliding window classifier and parallel instant impact detector
- Browser-based real-time dashboard with animated stickman, confidence ring, and live CSI waveform
- Automated Telegram fall alert with confidence score and class probability breakdown
- No-movement duration tracking for post-fall and prolonged stillness monitoring
- Privacy-preserving — no cameras, no wearables, only ambient WiFi signals

---

## Results

| Metric | Binary (Walk vs No Movement) | 3-Class (Fall + Walk + No Movement) |
|---|---|---|
| Test Accuracy | 100% | 100% |
| 5-Fold CV Accuracy | 98.9% +/- 1.1% | 99.1% +/- 1.8% |
| Dataset Size | 19,350 frames | 19,350 frames |
| Classes | 2 | 3 |

---

## System Architecture

```
ESP32 Microcontroller (52 subcarriers @ 43fps)
         |
         |  USB Serial (115200 baud)
         v
+------------------------------------------+
|            Python Backend                |
|                                          |
|  Preprocessing --> Feature Extraction    |
|       |                  |               |
|  Voting Ensemble    Impact Detector      |
|       |                  |               |
|       +------ Prediction ------+         |
+------------------------------------------+
         |                    |
         v                    v
  Browser Dashboard     Telegram Alert
  (Real-time UI)        (Caregiver)
```

---

## Project Structure

```
CSI-HAR-Project/
|
|-- dataset_raw/                    # Raw collected CSI data
|   |-- fall/                       # 15 recordings x ~430 frames
|   |-- walking/                    # 15 recordings x ~430 frames
|   +-- no_movement/                # 15 recordings x ~430 frames
|
|-- esp32 firmware/                 # Arduino firmware for ESP32
|
|-- models_3class/                  # Trained 3-class model files
|   |-- voting_clf.pkl
|   |-- scaler.pkl
|   +-- top_sc.npy
|
|-- models_binary/                  # Trained binary model files
|   |-- voting_clf.pkl
|   |-- scaler.pkl
|   +-- top_sc.npy
|
|-- binary_classifier.py            # Train Walking vs No Movement
|-- three_class_classifier.py       # Train Fall + Walking + No Movement
|-- realtime_ui_3class.py           # 3-class browser dashboard
|-- realtime_ui_2class.py           # Binary browser dashboard
|-- csi_dataset_recorder.py         # Record new CSI data
|-- realtime_predictor.py           # Terminal-based predictor
+-- README.md
```

---

## Hardware Setup

| Component | Details |
|---|---|
| Microcontroller | ESP32 with CSI support |
| Router | Any standard WiFi router |
| Distance | 2 to 3 metres Line-of-Sight |
| Connection | USB Serial at 115200 baud |
| Subcarriers | 52 |
| Frame Rate | ~43 fps |

```
[WiFi Router] <--------- 2-3m LOS ---------> [ESP32]
                               |
                         Activity Zone
                        (subject performs
                         activities here)
                               |
                          USB Serial
                               |
                           [Laptop]
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install pyserial scikit-learn scipy joblib numpy
```

### 2. Flash ESP32 Firmware

Upload the firmware from the `esp32 firmware/` folder to your ESP32 using Arduino IDE.

### 3. Record Dataset

Skip this step if using the existing dataset_raw folder.

```bash
python csi_dataset_recorder.py
```

### 4. Train Models

```bash
# Train binary classifier (Walking vs No Movement)
python binary_classifier.py

# Train 3-class classifier (Fall + Walking + No Movement)
python three_class_classifier.py
```

### 5. Run Real-time Dashboard

```bash
# 3-class dashboard (Fall + Walking + No Movement)
python realtime_ui_3class.py

# Binary dashboard (Walking + No Movement)
python realtime_ui_2class.py

# With manual port specification
python realtime_ui_3class.py --port COM3          # Windows
python realtime_ui_3class.py --port /dev/ttyUSB0  # Linux
```

Browser opens automatically. Click START to begin real-time prediction.

---

## Dashboard Features

| Feature | Description |
|---|---|
| Stickman Animation | Walking cycle, breathing idle, fall collapse animation |
| Confidence Ring | Animated SVG arc showing prediction confidence percentage |
| Live CSI Signal | Real-time waveform from ESP32 subcarriers |
| Confidence Trend | Sparkline chart of last 30 predictions |
| Activity Duration | Time spent in each activity during session |
| History Timeline | Last 20 predictions displayed as color-coded dots |
| Telegram Status | Live sending / sent / failed status indicator |
| No-Movement Timer | Tracks continuous stillness duration |
| Fall Overlay | Full-screen red alert with audio beep on fall detection |
| Session Timer | Live HH:MM:SS session clock |

---

## Telegram Alert Format

A professional fall notification is sent to the caregiver immediately upon fall detection.

```
FALL DETECTED
------------------------------
Time          : 2026-05-22 11:45:32
Detection     : ML Model (Voting Ensemble)
Frame No      : 1842
Confidence    : 87.3%
Probabilities : Fall 87% | Walking 8% | No Movement 5%
System        : Home CSI Monitor
------------------------------
Please check on the person immediately.
```

---

## Methodology

### Feature Engineering — 24 Features Per Subcarrier

Standard features (18):
Mean, Standard Deviation, Variance, Signal Energy, Amplitude Range, Mean Delta, Delta Variance, Max Delta, Skewness, Kurtosis, Zero Crossing Rate, Dominant Frequency Magnitude, Dominant Frequency, Spectral Entropy, Spectral Peak Ratio, Low Frequency Power, Mid Frequency Power, Autocorrelation

Fall-specific features (6):
Early Variance, Late Variance, Mid/Edge Variance Ratio, Variance Asymmetry, Max Delta Ratio, Energy Centroid Position

### Model — Soft Voting Ensemble

```python
VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=400)),
        ('gbm', GradientBoostingClassifier(n_estimators=200)),
        ('svm', SVC(kernel='rbf', probability=True))
    ],
    voting='soft'
)
```

### Sliding Window Parameters

| Parameter | Binary | 3-Class |
|---|---|---|
| Window Size | 80 frames | 80 frames |
| Step Size | 43 frames | 43 frames |
| Vote Window | 3 predictions | 3 predictions |

---

# Authors

V Jagan,
Siyana A,
Lisha John,
Nandana S

B.Tech Computer science,
College of Engineering and Management Punnapra Kerala,
Department of Computer Science and Engineering


---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- ESP32 CSI extraction based on open-source ESP32 WiFi CSI firmware
- scikit-learn library for machine learning pipeline
- Telegram Bot API for emergency fall notifications
