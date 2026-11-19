import os
import wave
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile

def extract_features(file_path, label):
    """Extracts Spectral, Phase Variance, and LSB statistical markers."""
    try:
        # --- 1. Raw Signal Features (LSB & Phase Domain) ---
        rate, raw_audio = wavfile.read(file_path)
        if len(raw_audio.shape) > 1:
            raw_audio = raw_audio[:, 0]

        # LSB Statistics (Catches Standard LSB in the first 8000 samples)
        sample_slice = raw_audio[:8000] if len(raw_audio) >= 8000 else raw_audio
        lsbs = sample_slice & 1
        lsb_mean = np.mean(lsbs)
        # Transition rate: how often consecutive LSBs flip between 0 and 1
        lsb_transitions = np.mean(np.abs(np.diff(lsbs)))

        # Phase Domain Statistics (Catches FFT Phase Coding)
        fft_data = np.fft.rfft(raw_audio)
        phase = np.angle(fft_data)
        # Focus on the embedding band (bins 100 to 1100) and overall phase diff
        band_phase = phase[100:1100] if len(phase) > 1100 else phase
        phase_diff = np.angle(np.exp(1j * np.diff(band_phase)))
        phase_var = np.var(phase_diff)
        phase_std = np.std(phase_diff)
        phase_mean = np.mean(np.abs(phase_diff))

        # --- 2. Librosa Spectral Features ---
        y, sr = librosa.load(file_path, sr=None)

        # MFCCs (Mean & Standard Deviation)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs.T, axis=0)
        mfcc_stds = np.std(mfccs.T, axis=0)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

        # Spectral Centroid & Rolloff
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean, cent_std = np.mean(cent), np.std(cent)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        roll_mean, roll_std = np.mean(rolloff), np.std(rolloff)

        # Spectral Entropy (with + 1e-10 fix to prevent division by zero)
        S, _ = librosa.magphase(librosa.stft(y))
        power_spectrum = np.abs(S) ** 2
        p = power_spectrum / (np.sum(power_spectrum, axis=0, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(p * np.log2(p + 1e-10), axis=0)
        entropy_mean, entropy_std = np.mean(spectral_entropy), np.std(spectral_entropy)

        # Assemble feature dictionary
        features = {
            'lsb_mean': lsb_mean,
            'lsb_transitions': lsb_transitions,
            'phase_var': phase_var,
            'phase_std': phase_std,
            'phase_mean': phase_mean,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'cent_mean': cent_mean,
            'cent_std': cent_std,
            'roll_mean': roll_mean,
            'roll_std': roll_std,
            'entropy_mean': entropy_mean,
            'entropy_std': entropy_std,
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
    print("Starting Phase 3: Advanced DLP Feature Extraction...")
    data = []

    # 0 = CLEAN, 1 = STANDARD_LSB, 2 = PHASE_ENCODED
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

            if index % 100 == 0 or index == len(files):
                print(f"  Extracted {index}/{len(files)} files...")

    df = pd.DataFrame(data)
    output_csv = os.path.join(base_dir, 'audio_features.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nPhase 3 Complete! Dataset saved to {output_csv}")

if __name__ == "__main__":
    base_data_directory = r"D:\FINAL_PROJECT\ISE\data"
    build_feature_dataset(base_data_directory)