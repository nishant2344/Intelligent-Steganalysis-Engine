import os
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.stats import kurtosis, skew

def extract_features(file_path, label):
    """Extracts Spectral, Byte-Level LSB, and FFT Phase-Ringing forensic markers."""
    try:
        rate, raw_audio = wavfile.read(file_path)
        if len(raw_audio.shape) > 1:
            raw_audio = raw_audio[:, 0]

        # --- 1. Byte-Level LSB Forensics (Catches Standard LSB) ---
        raw_bytes = raw_audio.view(np.uint8)
        head_bytes = raw_bytes[:8000] if len(raw_bytes) >= 8000 else raw_bytes
        tail_bytes = raw_bytes[8000:16000] if len(raw_bytes) >= 16000 else raw_bytes

        head_lsbs = head_bytes & 1
        tail_lsbs = tail_bytes & 1

        # Compare head (potentially infected) vs tail (clean)
        lsb_head_mean = np.mean(head_lsbs)
        lsb_tail_mean = np.mean(tail_lsbs)
        lsb_mean_diff = abs(lsb_head_mean - lsb_tail_mean)

        lsb_head_trans = np.mean(np.abs(np.diff(head_lsbs.astype(np.int8))))
        lsb_tail_trans = np.mean(np.abs(np.diff(tail_lsbs.astype(np.int8))))
        lsb_trans_diff = abs(lsb_head_trans - lsb_tail_trans)

        # Odd-byte LSBs (bit 8 of 16-bit PCM) - heavily affected by byte-level LSB
        odd_head_lsbs = head_bytes[1::2] & 1
        odd_tail_lsbs = tail_bytes[1::2] & 1
        odd_lsb_diff = abs(np.mean(odd_head_lsbs) - np.mean(odd_tail_lsbs))

        # --- 2. FFT Phase & Edge-Ringing Forensics (Catches Phase Coding) ---
        audio_float = raw_audio.astype(np.float64)
        total_rms = np.sqrt(np.mean(audio_float ** 2)) + 1e-10

        # Global FFT phase shifts leak energy into the silent start/end edges
        start_edge = audio_float[:500]
        end_edge = audio_float[-500:]
        start_rms_ratio = np.sqrt(np.mean(start_edge ** 2)) / total_rms
        end_rms_ratio = np.sqrt(np.mean(end_edge ** 2)) / total_rms
        start_max_ratio = np.max(np.abs(start_edge)) / (np.max(np.abs(audio_float)) + 1e-10)
        end_max_ratio = np.max(np.abs(end_edge)) / (np.max(np.abs(audio_float)) + 1e-10)

        # Phase derivative contrast: Target band (100:1100) vs Adjacent band (1100:2100)
        fft_data = np.fft.rfft(raw_audio)
        phase = np.angle(fft_data)

        band1 = phase[100:1100] if len(phase) > 1100 else phase
        band2 = phase[1100:2100] if len(phase) > 2100 else phase

        diff1 = np.angle(np.exp(1j * np.diff(band1)))
        diff2 = np.angle(np.exp(1j * np.diff(band2)))

        # Second-order phase difference (group delay)
        diff1_2nd = np.angle(np.exp(1j * np.diff(diff1)))

        phase_var_ratio = np.var(diff1) / (np.var(diff2) + 1e-10)
        phase_kurt_1 = kurtosis(diff1)
        phase_skew_1 = skew(diff1)
        phase_2nd_std = np.std(diff1_2nd)

        # --- 3. Librosa Spectral Features ---
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

        features['label'] = label
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_feature_dataset(base_dir):
    print("Starting Upgraded Forensic Feature Extraction...")
    data = []

    directories = [
        (os.path.join(base_dir, 'clean_wav'), 0),
        (os.path.join(base_dir, 'infected_wav'), 1),
        (os.path.join(base_dir, 'infected_wav_phase'), 2)
    ]

    for folder, label in directories:
        if not os.path.exists(folder):
            print(f"[!] Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"Processing {len(files)} files in {os.path.basename(folder)} (Label: {label})...")

        for index, filename in enumerate(files, 1):
            file_path = os.path.join(folder, filename)
            features = extract_features(file_path, label)
            if features:
                data.append(features)

            if index % 200 == 0 or index == len(files):
                print(f"  Extracted {index}/{len(files)} files...")

    df = pd.DataFrame(data)
    output_csv = os.path.join(base_dir, 'audio_features.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nExtraction Complete! Saved {len(df)} rows to {output_csv}")

if __name__ == "__main__":
    base_data_directory = r"D:\FINAL_PROJECT\ISE\data"
    build_feature_dataset(base_data_directory)