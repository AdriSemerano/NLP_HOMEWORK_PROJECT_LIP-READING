import os

# --- Configuration ---
# Path to the root of the new, processed GRID structure
new_grid_path = r"D:\Sapienza\NLP\New_GRID"

# --- Script Logic ---
if not os.path.isdir(new_grid_path):
    print(f"Error: The directory '{new_grid_path}' was not found.")
else:
    print("Processing .align files and rewriting their content...")

    # Walk through the directory tree starting from the new GRID root
    for root, dirs, files in os.walk(new_grid_path):
        for filename in files:
            # We are only interested in the .align files
            if filename.endswith('.align'):
                full_align_path = os.path.join(root, filename)

                # Read the contents of the align file
                words = []
                try:
                    with open(full_align_path, 'r') as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            # The word/label is the third element of the line
                            if len(parts) >= 3:
                                word = parts[2]
                                words.append(word)
                except Exception as e:
                    print(f"Error reading file {full_align_path}: {e}")
                    continue

                # Join the words into a single capitalized sentence
                sentence = ' '.join(words).upper()

                # Overwrite the original .align file with the new sentence
                try:
                    with open(full_align_path, 'w') as f_out:
                        f_out.write(sentence)
                    print(f"Successfully rewrote: {full_align_path}")
                except Exception as e:
                    print(f"Error writing to file {full_align_path}: {e}")

print("\nAll .align files have been processed.")