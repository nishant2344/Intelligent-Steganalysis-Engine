import numpy as np
import scipy.io.wavfile as wavfile
import os

def text_to_binary(text):
    """Converts a string to an 8-bit binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary_str):
    """Converts a binary string back to a text string."""
    text = ""
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if byte:
            text += chr(int(byte, 2))
    return text

def encode_phase(input_wav, output_wav, secret_message, start_bin=100):
    """Encodes a secret message into the audio phase using Real FFT."""
    print(f"Encoding message: '{secret_message}'")
    rate, audio_data = wavfile.read(input_wav)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    binary_msg = text_to_binary(secret_message)
    msg_length = len(binary_msg)
    
    # Use rfft to preserve real-signal conjugate symmetry automatically
    fft_data = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_data)
    phase = np.angle(fft_data)
    
    # Start at start_bin (100) to avoid DC component and unstable low bins
    for i in range(msg_length):
        idx = start_bin + i
        if binary_msg[i] == '0':
            phase[idx] += 0.25  # Clear positive shift
        else:
            phase[idx] -= 0.25  # Clear negative shift
            
    # Reconstruct using irfft and round cleanly before int16 conversion
    new_fft_data = magnitude * np.exp(1j * phase)
    new_audio_data = np.fft.irfft(new_fft_data, n=len(audio_data))
    
    wavfile.write(output_wav, rate, np.round(new_audio_data).astype(np.int16))
    print(f"File saved successfully to: {output_wav}")

def decode_phase(stego_wav, original_wav, msg_length_bits, start_bin=100):
    """Decodes the secret message using wrap-safe phase difference."""
    print("Extracting payload...")
    _, orig_audio = wavfile.read(original_wav)
    _, stego_audio = wavfile.read(stego_wav)
    
    if len(orig_audio.shape) > 1: orig_audio = orig_audio[:, 0]
    if len(stego_audio.shape) > 1: stego_audio = stego_audio[:, 0]
        
    orig_phase = np.angle(np.fft.rfft(orig_audio))
    stego_phase = np.angle(np.fft.rfft(stego_audio))
    
    extracted_bin = ""
    for i in range(msg_length_bits):
        idx = start_bin + i
        # Wrap-safe phase difference in range (-pi, pi]
        diff = np.angle(np.exp(1j * (stego_phase[idx] - orig_phase[idx])))
        if diff > 0:
            extracted_bin += '0'
        else:
            extracted_bin += '1'
            
    secret_message = binary_to_text(extracted_bin)
    print(f"\n[+] DECODED MESSAGE: {secret_message}")

if __name__ == "__main__":
    print("--- Enterprise DLP: Phase Coding PoC ---")
    
    # Keep your working file path here
    test_input = r"D:\FINAL_PROJECT\ISE\data\clean_wav\audio_001.wav" 
    test_output = r"D:\FINAL_PROJECT\ISE\data\phase_encoded_poc.wav"
    secret_text = "Alpha Test 001 helo paji"
    
    if os.path.exists(test_input):
        encode_phase(test_input, test_output, secret_text)
        bit_length = len(secret_text) * 8
        decode_phase(test_output, test_input, bit_length)
    else:
        print(f"[!] Error: Could not find {test_input}. Please check the filename.")