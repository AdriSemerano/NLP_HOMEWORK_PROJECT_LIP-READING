import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import git
import shutil

# -------- CONFIG --------
LRS2_PATH = ""
OUTPUT_PATH = "output_clips"
REPO_PATH = "Lipreading_using_Temporal_Convolutional_Networks"
REPO_URL = "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git"
LABELS_PATH = r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\labels\500WordsSortedList.txt"
# ------------------------

def ask_for_path(prompt, default=None):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def clone_repo():
    if not os.path.exists(REPO_PATH):
        print(f"Cloning repo into {REPO_PATH}...")
        git.Repo.clone_from(REPO_URL, REPO_PATH)
        print("Repository cloned.")
    else:
        print("Repository already exists. Skipping cloning.")

def load_allowed_words(txt_path):
    with open(txt_path, 'r') as f:
        return set(word.strip().upper() for word in f.readlines())

def extract_word_segments(txt_file):
    segments = []
    with open(txt_file, "r") as f:
        lines = f.readlines()
    parsing = False
    for line in lines:
        if line.startswith("WORD "):
            parsing = True
            continue
        if parsing:
            parts = line.strip().split()
            if len(parts) >= 3:
                word, start, end = parts[0].upper(), float(parts[1]), float(parts[2])
                segments.append((word, start, end))
    return segments

def extract_video_segment(video_path, word, clip_id, start, end, output_root):
    output_dir = os.path.join(output_root, word)
    os.makedirs(output_dir, exist_ok=True)
    output_clip = os.path.join(output_dir, f"{clip_id}.mp4")
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-y", output_clip
    ]
    subprocess.run(cmd)

def split_dataset(lrs2_root, output_root):
    allowed_words = load_allowed_words(LABELS_PATH)

    lrs2_root = Path(lrs2_root)
    output_root = Path(output_root)

    if not lrs2_root.exists():
        print(f"Dataset path not found: {lrs2_root}")
        return

    folders = [f for f in lrs2_root.rglob("*") if f.is_dir()]

    video_txt_pairs = []
    for folder in folders:
        mp4_files = list(folder.glob("*.mp4"))
        for mp4_file in mp4_files:
            stem = mp4_file.stem
            txt_file = folder / f"{stem}.txt"
            if txt_file.exists():
                video_txt_pairs.append((mp4_file, txt_file))

    print(f"Found {len(video_txt_pairs)} pairs video+txt...")

    total_extracted = 0

    for video_path, txt_path in tqdm(video_txt_pairs, desc="Processing video segments"):
        base = video_path.stem
        segments = extract_word_segments(txt_path)
        for idx, (word, start, end) in enumerate(segments):
            if word.upper() not in allowed_words:
                continue  # ⚠️ ignores unknown words by the model
            clip_id = f"{base}_{word}_{idx}"
            extract_video_segment(video_path, word, clip_id, start, end, output_root)
            total_extracted += 1

    print(f"\n Number generated clips: {total_extracted}")

if __name__ == "__main__":
    print("==== LRS2 Splitter & Repo Cloner ====")

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        print(f"Folder '{OUTPUT_PATH}' removed.")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    clone_repo()

    if LRS2_PATH == "":
        LRS2_PATH = ask_for_path("Path_Dataset", "./LRS2")
    if not os.path.exists(LRS2_PATH):
        print(f"Invalid Path: {LRS2_PATH}")
        exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    split_dataset(LRS2_PATH, OUTPUT_PATH)

    print(f"\n Operation Complete. Clips saved in: {OUTPUT_PATH}")
