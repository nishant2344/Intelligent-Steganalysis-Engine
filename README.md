# 🛡️ Zero-Trust Media DLP Firewall & Phase Steganography Suite (ISE)

An enterprise-grade **Audio Steganalysis & Data Loss Prevention (DLP)** suite built in Python. Moving beyond fragile Least Significant Bit (LSB) steganography, this project implements **FFT-based Phase Coding** for covert audio embedding and deploys an **XGBoost multi-class detection engine** (83.29% accuracy) wrapped in an interactive Streamlit DLP firewall dashboard.

---

## ✨ Key Capabilities

* **Red Team — FFT Phase-Coding Engine:** Embeds secret payloads into the frequency-domain phase angles of `.wav` files using Real Fast Fourier Transform (`rfft` / `irfft`), leaving audio amplitude untouched and largely imperceptible to human hearing.
* **Blue Team — Forensic Feature Extractor:** Extracts 39 targeted statistical, spectral, and signal-processing markers, including:
  * Byte-level LSB head-vs-tail transition differentials (catches Standard LSB at **99% F1-score**).
  * Global FFT edge-ringing RMS/max ratios and second-order phase derivatives (group delay) to detect phase manipulation.
  * MFCCs (mean & standard deviation), Zero-Crossing Rate, Spectral Centroid, Spectral Rolloff, and Spectral Entropy.
* **Multi-Class ML Detection Engine:** Classifies audio traffic into `[CLEAN]`, `[STANDARD_LSB]`, or `[PHASE_ENCODED]` using a tuned **XGBoost** classifier.
* **Interactive DLP Dashboard:** A two-tab Streamlit web app for live audio scanning (Blue Team) and live phase-encoded payload generation (Red Team).

---

## 📊 Model Performance (XGBoost)

Evaluated on a balanced multi-class dataset (8,109 `.wav` files across 3 classes):

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **CLEAN (0)** | 0.74 | 0.79 | 0.76 |
| **STANDARD_LSB (1)** | 0.99 | 0.99 | 0.99 |
| **PHASE_ENCODED (2)** | 0.78 | 0.71 | 0.74 |
| **Overall Accuracy** | **83.29%** | | |

---

## 📁 Project Structure

```text
ISE/
├── app.py                        # Streamlit Enterprise DLP Dashboard (Blue & Red Team UI)
├── phase_stego_mini.py           # Standalone FFT Phase Coding Proof of Concept (Encode/Decode)
├── generate_phase_dataset.py     # Bulk Phase-Encoded .wav dataset generator
├── extractor.py                  # 39-feature forensic extraction pipeline -> audio_features.csv
├── train_model.py                # Baseline Random Forest training script
├── tune_model.py                 # Model benchmarking script (XGBoost, HistGradientBoosting, ExtraTrees)
├── requirements.txt              # Python dependencies
├── models/
│   └── stego_detector_model.joblib  # Saved trained XGBoost model
└── data/                         # Local audio datasets & extracted CSV (excluded from Git)
    ├── clean_wav/
    ├── infected_wav/
    └── infected_wav_phase/
```

---

## 🚀 Local Setup Guide (Step-by-Step)

Follow these steps to clone and run the project on your local machine.

### Prerequisites
* **Python 3.9+** installed on your system
* **Git** installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create & Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt / Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Local Paths (First-Time Setup)
Before running the scripts locally, make sure the file paths at the top/bottom of `app.py`, `extractor.py`, and `tune_model.py` point to your local project directory (for example, update `MODEL_PATH` in `app.py` to `os.path.join("models", "stego_detector_model.joblib")`).

### 5. Launch the Enterprise DLP Dashboard
```bash
streamlit run app.py
```
Your browser will automatically open the dashboard at `http://localhost:8501`.

---

## 🧪 Running the Full Pipeline from Scratch (Optional)

If you want to reproduce the dataset synthesis, feature extraction, and model training locally:

1. **Test the standalone Phase Coding PoC:**
   Place a clean `.wav` file in `data/clean_wav/`, update `test_input` in `phase_stego_mini.py`, and run:
   ```bash
   python phase_stego_mini.py
   ```
2. **Generate the Phase-Encoded dataset:**
   ```bash
   python generate_phase_dataset.py
   ```
3. **Extract the 39 forensic features into `data/audio_features.csv`:**
   ```bash
   python extractor.py
   ```
4. **Train and benchmark models (saves the winning model to `models/`):**
   ```bash
   python tune_model.py
   ```

---

## 🛠️ Tech Stack
* **Signal Processing (DSP):** `numpy`, `scipy`, `librosa`
* **Machine Learning:** `xgboost`, `scikit-learn`, `pandas`, `joblib`
* **Frontend / Dashboard:** `streamlit`