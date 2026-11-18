import wave
import os
import random
import numpy as np

def embed_standard_lsb(input_wav, output_wav, secret_data_bits):
    """Standard sequential LSB embedding (The Baseline)."""
    with wave.open(input_wav, mode='rb') as audio:
        frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))
        params = audio.getparams()
    
    if len(secret_data_bits) > len(frame_bytes):
        raise ValueError(f"Audio file {input_wav} too small for payload.")
        
    for i in range(len(secret_data_bits)):
        frame_bytes[i] = (frame_bytes[i] & 254) | secret_data_bits[i]
        
    with wave.open(output_wav, 'wb') as infected_audio:
        infected_audio.setparams(params)
        infected_audio.writeframes(frame_bytes)

def embed_custom_stego(input_wav, output_wav, secret_data_bits, seed=999):
    """Proprietary Dynamic Seed-Based Steganography (Placeholder for Encryption)."""
    with wave.open(input_wav, mode='rb') as audio:
        frame_bytes = bytearray(list(audio.readframes(audio.getnframes())))
        params = audio.getparams()
        
    if len(secret_data_bits) > len(frame_bytes):
        raise ValueError(f"Audio file {input_wav} too small for payload.")
    
    # Use a seed to generate a pseudo-random sequence of frame indices
    random.seed(seed)
    indices = random.sample(range(len(frame_bytes)), len(secret_data_bits))
    
    # Scatter the bits across the random indices
    for i, bit in enumerate(secret_data_bits):
        target_index = indices[i]
        frame_bytes[target_index] = (frame_bytes[target_index] & 254) | bit
        
    with wave.open(output_wav, 'wb') as infected_audio:
        infected_audio.setparams(params)
        infected_audio.writeframes(frame_bytes)

def generate_multi_class_dataset(base_dir):
    """Generates both standard and custom infected files for the multi-class model."""
    clean_folder = os.path.join(base_dir, 'clean_wav')
    lsb_folder = os.path.join(base_dir, 'infected_wav')
    custom_folder = os.path.join(base_dir, 'infected_wav_custom')

    for folder in [lsb_folder, custom_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]
    total_files = len(files)
    print(f"Found {total_files} .wav files. Generating multi-class datasets...")
    
    for index, filename in enumerate(files, 1):
        clean_path = os.path.join(clean_folder, filename)
        lsb_path = os.path.join(lsb_folder, "lsb_" + filename)
        custom_path = os.path.join(custom_folder, "custom_" + filename)
        
        # Generate a random payload (8000 bits)
        random_payload = [random.randint(0, 1) for _ in range(8000)]
        
        try:
            embed_standard_lsb(clean_path, lsb_path, random_payload)
            # Use a secret key seed (e.g., 999) for your proprietary method
            embed_custom_stego(clean_path, custom_path, random_payload, seed=999)
            
            if index % 100 == 0 or index == total_files:
                print(f"Processed {index}/{total_files} files...")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    print("Starting Multi-Class Data Generation...")
    # Pointing exactly to your folder structure
    base_data_directory = r"D:\FINAL_PROJECT\ISE\data"
    generate_multi_class_dataset(base_data_directory)
    print("Data Generation Complete.")