import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path, label):
    """Extracts MFCCs and Spectral Entropy from an audio file."""
    try:
        # Load audio file (librosa automatically converts to mono and normalizes)
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        # We take the mean of each of the 13 coefficients across time
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs.T, axis=0)
        
        # 2. Extract Spectral Entropy
        # Calculate the power spectrum
        S, phase = librosa.magphase(librosa.stft(y))
        power_spectrum = np.abs(S)**2
        # Normalize to get a probability distribution
        p = power_spectrum / np.sum(power_spectrum, axis=0, keepdims=True)
        # Calculate entropy
        spectral_entropy = -np.sum(p * np.log2(p + 1e-10), axis=0)
        mean_spectral_entropy = np.mean(spectral_entropy)
        
        # Combine features into a single dictionary
        features = {f'mfcc_{i+1}': mfcc_means[i] for i in range(13)}
        features['spectral_entropy'] = mean_spectral_entropy
        features['label'] = label
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_feature_dataset(base_dir):
    print("Starting feature extraction... This will take a while.")
    data = []
    
    # Define folders and their corresponding labels
    # 0 = Clean, 1 = Standard LSB, 2 = Custom Stego
    directories = [
        (os.path.join(base_dir, 'clean_wav'), 0),
        (os.path.join(base_dir, 'infected_wav'), 1),
        (os.path.join(base_dir, 'infected_wav_custom'), 2)
    ]
    
    for folder, label in directories:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
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
                
    # Save to a structured CSV for the Machine Learning model
    df = pd.DataFrame(data)
    output_csv = os.path.join(base_dir, 'audio_features.csv')
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Dataset saved to {output_csv}")

if __name__ == "__main__":
    # Pointing exactly to your data folder structure
    base_data_directory = r"D:\FINAL_PROJECT\ISE\data"
    build_feature_dataset(base_data_directory)