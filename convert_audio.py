import os
import glob
import soundfile as sf

# Define paths relative to your ISE directory
source_dir = os.path.join("data", "LibriSpeech")
target_dir = os.path.join("data", "clean_wav")

# Recursively find all .flac files in the LibriSpeech directory
flac_files = glob.glob(os.path.join(source_dir, "**", "*.flac"), recursive=True)
total_files = len(flac_files)
print(f"Found {total_files} .flac files in total.")
print(f"Starting conversion of all {total_files} files to .wav...")

count = 0

for file_path in flac_files:
    # Extract the original filename and change the extension
    base_name = os.path.basename(file_path)
    wav_name = os.path.splitext(base_name)[0] + ".wav"
    target_path = os.path.join(target_dir, wav_name)
    
    # Read the .flac file and write it as .wav
    data, samplerate = sf.read(file_path)
    sf.write(target_path, data, samplerate)
    
    count += 1
    # Print progress every 500 files to keep the terminal clean
    if count % 500 == 0 or count == total_files:
        print(f"Converted {count}/{total_files} files...")

print(f"Conversion complete! Check your '{target_dir}' folder.")