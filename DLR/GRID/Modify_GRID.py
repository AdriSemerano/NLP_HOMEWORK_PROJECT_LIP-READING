import os
import shutil

# --- Configuration ---
# Path to the original GRID corpus
grid_corpus_path = r"D:\Sapienza\NLP\GRID"

# Path where the new, reorganized structure will be created
new_grid_path = r"D:\Sapienza\NLP\New_GRID"

# --- Script Logic ---

if not os.path.exists(new_grid_path):
    os.makedirs(new_grid_path)
    print(f"Created new directory: {new_grid_path}")
else:
    print(f"Directory already exists: {new_grid_path}. Files will be added or overwritten.")

# Get a list of all speaker directories in the original GRID corpus
speaker_dirs = [d for d in os.listdir(grid_corpus_path) if os.path.isdir(os.path.join(grid_corpus_path, d))]

print(f"Found {len(speaker_dirs)} speaker directories.")

for speaker_dir in speaker_dirs:
    source_speaker_path = os.path.join(grid_corpus_path, speaker_dir)
    dest_speaker_path = os.path.join(new_grid_path, speaker_dir)

    # Create the new speaker folder if it doesn't exist
    if not os.path.exists(dest_speaker_path):
        os.makedirs(dest_speaker_path)

    # --- Copy video files ---
    for file in os.listdir(source_speaker_path):
        if file.endswith('.mpg'):
            source_file = os.path.join(source_speaker_path, file)
            dest_file = os.path.join(dest_speaker_path, file)
            shutil.copy2(source_file, dest_file)

    # --- Copy alignment files ---
    align_path = os.path.join(source_speaker_path, 'align')
    if os.path.exists(align_path):
        for file in os.listdir(align_path):
            source_file = os.path.join(align_path, file)
            dest_file = os.path.join(dest_speaker_path, file)
            shutil.copy2(source_file, dest_file)

print("\nReorganization complete.")
print(f"The new structure is located at: {new_grid_path}")