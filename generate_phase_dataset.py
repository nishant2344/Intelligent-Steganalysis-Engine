import os
import random
import numpy as np
import scipy.io.wavfile as wavfile

def embed_phase_bulk(input_wav, output_wav, num_bits=1000, start_bin=100):
    """Embeds a random binary payload into the audio phase using Real FFT."""
    rate, audio_data = wavfile.read(input_wav)
    
    # Force mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    fft_data = np.fft.rfft(audio_data)
    
    # Ensure the audio file has enough frequency bins for the payload
    if start_bin + num_bits >= len(fft_data):
        raise ValueError(f"Audio file {input_wav} too short for {num_bits}-bit phase payload.")
        
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # Generate random bits and shift the phase angles
    bits = np.random.randint(0, 2, size=num_bits)
    shifts = np.where(bits == 0, 0.25, -0.25)
    phase[start_bin : start_bin + num_bits] += shifts
    
    # Reconstruct audio using Inverse Real FFT
    new_fft_data = magnitude * np.exp(1j * phase)
    new_audio_data = np.fft.irfft(new_fft_data, n=len(audio_data))
    
    wavfile.write(output_wav, rate, np.round(new_audio_data).astype(np.int16))

def build_phase_dataset(base_dir):
    clean_folder = os.path.join(base_dir, 'clean_wav')
    phase_folder = os.path.join(base_dir, 'infected_wav_phase')
    
    if not os.path.exists(phase_folder):
        os.makedirs(phase_folder)
        
    files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]
    total_files = len(files)
    print(f"Found {total_files} clean files. Starting Phase-Encoded bulk generation...")
    
    for index, filename in enumerate(files, 1):
        clean_path = os.path.join(clean_folder, filename)
        phase_path = os.path.join(phase_folder, "phase_" + filename)
        
        try:
            embed_phase_bulk(clean_path, phase_path, num_bits=1000, start_bin=100)
            if index % 100 == 0 or index == total_files:
                print(f"Processed {index}/{total_files} files...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print("Phase-Encoded Dataset Generation Complete!")

if __name__ == "__main__":
    base_data_directory = r"D:\FINAL_PROJECT\ISE\data"
    build_phase_dataset(base_data_directory)