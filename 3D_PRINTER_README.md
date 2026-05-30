# 🖨️ 3D Printer Quality Analysis & Temperature Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-TensorFlow-red.svg)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project exploring two problems in 3D printing:

1. **Material Classification** — predicting whether a printed part used ABS or PLA filament from printer settings and output quality metrics, comparing KNN, SVM, and a Neural Network
2. **Temperature Forecasting** — predicting nozzle temperature over time from real 3D printer sensor data using a stacked LSTM

---

## 📋 Table of Contents

- [Results](#-results)
- [Datasets](#-datasets)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Findings](#-key-findings)
- [Technologies](#-technologies)

---

## 📊 Results

### Part 1 — Material Classification (ABS vs PLA)

| Model | Notes |
|-------|-------|
| **KNN** | Optimal k selected by sweeping k=1–14 on test accuracy |
| **SVM** | Linear kernel; evaluated with precision, recall, F1 |
| **Neural Network** | 3-layer MLP with BatchNorm + Dropout; trained for 100 epochs |

### Part 2 — Temperature Forecasting

| Model | Architecture | Metric |
|-------|-------------|--------|
| **Stacked LSTM** | 4 LSTM layers (50 units each) + Dropout | RMSE & MAE on held-out test set |

---

## 🗂 Datasets

### Dataset 1 — 3D Printer Settings & Quality
- **Source:** [Kaggle — 3D Printer Dataset](https://www.kaggle.com/datasets/afumetto/3dprinter)
- **Size:** 50 print jobs × 11 features
- **Target:** Material type (ABS = 0, PLA = 1)
- **Features:** layer height, wall thickness, infill density, infill pattern, nozzle temperature, bed temperature, print speed, fan speed, roughness, tension strength, elongation

### Dataset 2 — Real Sensor Data
- **Source:** Real 3D printer IoT sensor readings (`realdata.csv`)
- **Columns:** `createdAt` (ISO timestamp), `nozzleTemp`, `bedTemp`
- **Used for:** LSTM time-series forecasting of nozzle temperature

---

## 🧠 Models

### Material Classification

**KNN (K-Nearest Neighbours)**
- Features normalised to [0, 1] before training
- Optimal k found by evaluating k = 1 to 14 on the test set
- Simple, interpretable baseline

**SVM (Support Vector Machine)**
- Linear kernel
- Same normalised feature set as KNN
- Generally stronger generalisation on small datasets

**Neural Network (MLP)**
- Architecture: Dense(32) → BatchNorm → ReLU → Dropout(0.25) → Dense(64) → ReLU → Dropout(0.25) → Dense(2, softmax)
- Optimiser: Adam | Loss: Sparse Categorical Crossentropy
- Training/validation curves plotted per epoch

### Temperature Forecasting

**Stacked LSTM**
- Input: sliding window of 60 consecutive temperature readings
- Architecture: 4 × LSTM(50) with Dropout(0.2) between each layer → Dense(1)
- Target: next nozzle temperature value
- Optimiser: Adam | Loss: MSE
- Output scaled back to °C using MinMaxScaler inverse transform

---

## 📁 Project Structure

```
3d-printer-ml/
├── 3D_Printer_ML_Analysis.ipynb   # Main notebook (all models)
├── README.md
└── data/                          # Add your CSV files here
    ├── 3D_printer.csv             # Kaggle dataset
    └── realdata.csv               # Sensor data
```

---

## 🚀 How to Run

**Option A — Google Colab (recommended)**

1. Open `3D_Printer_ML_Analysis.ipynb` in Google Colab
2. Upload `3D_printer.csv` and `realdata.csv` to the Colab session files (or mount Drive)
3. Update the two path variables at the top of the notebook:
```python
DATA_PATH   = '3D_printer.csv'   # path to printer settings dataset
SENSOR_PATH = 'realdata.csv'     # path to sensor data
```
4. Runtime → Run all

**Option B — Local**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
jupyter notebook 3D_Printer_ML_Analysis.ipynb
```

---

## 🔍 Key Findings

1. **ABS and PLA are linearly separable in most feature subspaces** — both KNN and SVM achieve strong accuracy on the test set with minimal tuning.

2. **Fan speed negatively impacts ABS tension strength** — ABS is sensitive to cooling; excessive fan speed during printing reduces mechanical strength. PLA shows the opposite trend.

3. **Layer height and roughness are positively correlated for ABS** — as layer height increases, surface roughness increases more sharply for ABS than PLA, which prints more smoothly.

4. **The constraint-based fault diagnosis** correctly flags over-extrusion, under-extrusion, and warping risk based on predicted material type and printer settings — without requiring labelled fault data.

5. **The LSTM captures periodic temperature patterns** in the sensor data after 50 epochs, though performance depends heavily on dataset size and sequence regularity.

---

## 🛠 Technologies

`Python` · `Keras / TensorFlow` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Google Colab`

---

*Personal ML portfolio project — University of Stirling*
