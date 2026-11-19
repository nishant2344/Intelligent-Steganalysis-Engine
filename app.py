import os
import tempfile
import joblib
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.stats import kurtosis, skew
import streamlit as st

st.set_page_config(
    page_title="Zero-Trust Media DLP Firewall",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = r"D:\FINAL_PROJECT\ISE\models\stego_detector_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def extract_single_file_features(file_path):
    """Extracts the exact forensic feature vector expected by the XGBoost model."""
    rate, raw_audio = wavfile.read(file_path)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[:, 0]

    # 1. Byte-Level LSB Forensics
    raw_bytes = raw_audio.view(np.uint8)
    head_bytes = raw_bytes[:8000] if len(raw_bytes) >= 8000 else raw_bytes
    tail_bytes = raw_bytes[8000:16000] if len(raw_bytes) >= 16000 else raw_bytes

    head_lsbs = head_bytes & 1
    tail_lsbs = tail_bytes & 1

    lsb_head_mean = np.mean(head_lsbs)
    lsb_tail_mean = np.mean(tail_lsbs)
    lsb_mean_diff = abs(lsb_head_mean - lsb_tail_mean)

    lsb_head_trans = np.mean(np.abs(np.diff(head_lsbs.astype(np.int8))))
    lsb_tail_trans = np.mean(np.abs(np.diff(tail_lsbs.astype(np.int8))))
    lsb_trans_diff = abs(lsb_head_trans - lsb_tail_trans)

    odd_head_lsbs = head_bytes[1::2] & 1
    odd_tail_lsbs = tail_bytes[1::2] & 1
    odd_lsb_diff = abs(np.mean(odd_head_lsbs) - np.mean(odd_tail_lsbs))

    # 2. FFT Phase & Edge-Ringing Forensics
    audio_float = raw_audio.astype(np.float64)
    total_rms = np.sqrt(np.mean(audio_float ** 2)) + 1e-10

    start_edge = audio_float[:500]
    end_edge = audio_float[-500:]
    start_rms_ratio = np.sqrt(np.mean(start_edge ** 2)) / total_rms
    end_rms_ratio = np.sqrt(np.mean(end_edge ** 2)) / total_rms
    start_max_ratio = np.max(np.abs(start_edge)) / (np.max(np.abs(audio_float)) + 1e-10)
    end_max_ratio = np.max(np.abs(end_edge)) / (np.max(np.abs(audio_float)) + 1e-10)

    fft_data = np.fft.rfft(raw_audio)
    phase = np.angle(fft_data)

    band1 = phase[100:1100] if len(phase) > 1100 else phase
    band2 = phase[1100:2100] if len(phase) > 2100 else phase

    diff1 = np.angle(np.exp(1j * np.diff(band1)))
    diff2 = np.angle(np.exp(1j * np.diff(band2)))
    diff1_2nd = np.angle(np.exp(1j * np.diff(diff1)))

    phase_var_ratio = np.var(diff1) / (np.var(diff2) + 1e-10)
    phase_kurt_1 = kurtosis(diff1)
    phase_skew_1 = skew(diff1)
    phase_2nd_std = np.std(diff1_2nd)

    # 3. Librosa Spectral Features
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs.T, axis=0)
    mfcc_stds = np.std(mfccs.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    S, _ = librosa.magphase(librosa.stft(y))
    power_spectrum = np.abs(S) ** 2
    p = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(p * np.log2(p + 1e-10), axis=0)

    features = {
        'lsb_head_mean': lsb_head_mean,
        'lsb_mean_diff': lsb_mean_diff,
        'lsb_head_trans': lsb_head_trans,
        'lsb_trans_diff': lsb_trans_diff,
        'odd_lsb_diff': odd_lsb_diff,
        'start_rms_ratio': start_rms_ratio,
        'end_rms_ratio': end_rms_ratio,
        'start_max_ratio': start_max_ratio,
        'end_max_ratio': end_max_ratio,
        'phase_var_ratio': phase_var_ratio,
        'phase_kurt_1': phase_kurt_1,
        'phase_skew_1': phase_skew_1,
        'phase_2nd_std': phase_2nd_std,
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'cent_mean': np.mean(cent),
        'cent_std': np.std(cent),
        'roll_mean': np.mean(rolloff),
        'roll_std': np.std(rolloff),
        'entropy_mean': np.mean(spectral_entropy),
        'entropy_std': np.std(spectral_entropy),
    }
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = mfcc_means[i]
        features[f'mfcc_std_{i+1}'] = mfcc_stds[i]

    return pd.DataFrame([features])

def encode_phase_audio(input_path, output_path, secret_message, start_bin=100):
    """Encodes a text payload into the audio frequency phase using rfft/irfft."""
    rate, audio_data = wavfile.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    binary_msg = ''.join(format(ord(c), '08b') for c in secret_message)
    fft_data = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)

    for i, bit in enumerate(binary_msg):
        idx = start_bin + i
        phase[idx] += 0.25 if bit == '0' else -0.25

    new_audio = np.fft.irfft(magnitude * np.exp(1j * phase), n=len(audio_data))
    wavfile.write(output_path, rate, np.round(new_audio).astype(np.int16))

# --- UI Layout ---
st.title("🛡️ Zero-Trust Media DLP Firewall & Phase Steganography Suite")
st.caption("Enterprise Audio Steganalysis Engine | XGBoost Multi-Class Detector (83.29% Accuracy)")

tab1, tab2 = st.tabs(["🔍 Blue Team: DLP Firewall Scanner", "🧪 Red Team: Phase-Coding Studio"])

with tab1:
    st.subheader("Inspect Outgoing Audio Traffic")
    uploaded_file = st.file_uploader("Upload a .wav file for forensic inspection", type=["wav"], key="scanner")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Run DLP Deep Scan", type="primary"):
            with st.spinner("Extracting Byte-Level LSB, Phase-Ringing, and Spectral markers..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    model = load_model()
                    features_df = extract_single_file_features(tmp_path)
                    pred = int(model.predict(features_df)[0])
                    probs = model.predict_proba(features_df)[0]

                    labels = {
                        0: ("CLEAN", "success", "No covert payload detected. File cleared for network transit."),
                        1: ("STANDARD_LSB", "error", "ALERT: Byte-level LSB amplitude manipulation detected. Transit blocked."),
                        2: ("PHASE_ENCODED", "warning", "ALERT: Frequency-domain FFT phase anomaly detected. Transit blocked.")
                    }

                    verdict, status_type, msg = labels[pred]
                    getattr(st, status_type)(f"**VERDICT: [{verdict}]** — {msg}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Clean Confidence", f"{probs[0]*100:.1f}%")
                    col2.metric("Standard LSB Risk", f"{probs[1]*100:.1f}%")
                    col3.metric("Phase-Encoded Risk", f"{probs[2]*100:.1f}%")

                    with st.expander("View Extracted Forensic Markers"):
                        st.dataframe(features_df)
                finally:
                    os.remove(tmp_path)

with tab2:
    st.subheader("Generate Phase-Encoded Stego Audio")
    clean_upload = st.file_uploader("Upload a clean .wav file", type=["wav"], key="encoder")
    secret_msg = st.text_input("Secret Payload to Embed", value="Confidential Project Alpha")

    if clean_upload is not None and secret_msg:
        if st.button("Encode into Audio Phase"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                tmp_in.write(clean_upload.read())
                in_path = tmp_in.name

            out_path = in_path.replace(".wav", "_phase_stego.wav")
            try:
                encode_phase_audio(in_path, out_path, secret_msg)
                with open(out_path, "rb") as f:
                    stego_bytes = f.read()

                st.success(f"Embedded {len(secret_msg)*8} bits into the FFT phase matrix.")
                st.audio(stego_bytes, format="audio/wav")
                st.download_button(
                    label="Download Phase-Encoded .wav",
                    data=stego_bytes,
                    file_name="phase_encoded_payload.wav",
                    mime="audio/wav"
                )
            finally:
                if os.path.exists(in_path): os.remove(in_path)
                if os.path.exists(out_path): os.remove(out_path)