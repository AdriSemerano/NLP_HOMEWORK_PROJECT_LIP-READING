import urllib.request
import bz2
import shutil
import os
import cv2
import dlib
import numpy as np
import pickle
from tqdm import tqdm


URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
BZ2_FILE = "shape_predictor_68_face_landmarks.dat.bz2"
DAT_FILE = "shape_predictor_68_face_landmarks.dat"

output_root = r"Path_To_Project\output_clips"
landmark_root = r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\landmarks"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"

ANNOTATION_DIR = r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\annotations"
ANNOTATION_FILE = os.path.join(ANNOTATION_DIR, "annotations.txt")
BASE_DIR = r"Path_To_Project" 
TEST_CSV = os.path.join(BASE_DIR, "Lipreading_using_Temporal_Convolutional_Networks", "test.csv")


if not os.path.exists(DAT_FILE):
    print("Download file...")
    urllib.request.urlretrieve(URL, BZ2_FILE)

    print("Extraction .bz2...")
    with bz2.BZ2File(BZ2_FILE) as f_in, open(DAT_FILE, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(BZ2_FILE)
    print("File Available:", DAT_FILE)
else:
    print("File already downloaded:", DAT_FILE)

def clean_video_remove_no_faces(video_path):
    face_detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Can't open the video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    temp_path = video_path + ".tmp.mp4"
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    frame_idx = 0
    kept_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        if len(faces) > 0:
            out.write(frame)
            kept_frames += 1
        else:
            print(f"[FRAME {frame_idx}] No faces, frame skipped")

        frame_idx += 1

    cap.release()
    out.release()

    os.replace(temp_path, video_path)
    print(f"[VIDEO CLEAN] {video_path} clean: {kept_frames}/{frame_idx} frames kept")


def save_pts_file(points, path):
    with open(path, "w") as f:
        f.write(f"{len(points)}\n")
        for x, y in points:
            f.write(f"{x} {y}\n")


def save_landmarks_to_pkl(landmarks, path):
    with open(path, "wb") as f:
        pickle.dump(landmarks, f)


def write_annotation_file(path, samples):
    with open(path, "w") as f:
        for rel_path, label_idx, speaker in samples:
            rel_path = rel_path.replace(".mp4", "")
            f.write(f"{rel_path} {label_idx} {speaker} 0 29\n")
    print(f"[ANNOTATION] Written {path} with {len(samples)} lines")


def get_last_processed_video():
    """In case execution stops and need to restart from the last processed video"""
    if not os.path.exists(landmark_root):
        return None, None

    labels = sorted([d for d in os.listdir(landmark_root) if os.path.isdir(os.path.join(landmark_root, d))])
    if not labels:
        return None, None
    last_label = labels[-1]

    pkls = sorted([f for f in os.listdir(os.path.join(landmark_root, last_label)) if f.endswith(".pkl")])
    if not pkls:
        return last_label, None
    last_video = os.path.splitext(pkls[-1])[0]
    return last_label, last_video


#EXTRACT FACE LANDMARKS FROM VIDEOS

face_detector = dlib.get_frontal_face_detector()
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Model not found: {shape_predictor_path}")
landmark_predictor = dlib.shape_predictor(shape_predictor_path)

os.makedirs(landmark_root, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)


last_label, last_video = get_last_processed_video()
resume_mode = last_label is not None and last_video is not None
start_processing = not resume_mode

print(f"[RESUME] Last label: {last_label}, Last video: {last_video}")

print(f"[LANDMARKS] Start extraction from {output_root}")
for label in sorted(os.listdir(output_root)):
    label_path = os.path.join(output_root, label)
    if not os.path.isdir(label_path):
        continue

    for video_file in sorted(os.listdir(label_path)):
        if not video_file.endswith(".mp4"):
            continue

        if resume_mode and not start_processing:
            if label == last_label and os.path.splitext(video_file)[0] == last_video:
                start_processing = True
            continue

        video_path = os.path.join(label_path, video_file)
        video_name = os.path.splitext(video_file)[0]

        print(f"\n[VIDEO] {video_path}")
        clean_video_remove_no_faces(video_path)

        pts_dir = os.path.join(landmark_root, label, video_name)
        pkl_path = os.path.join(landmark_root, label, f"{video_name}.pkl")
        os.makedirs(pts_dir, exist_ok=True)

        start_frame = 0
        all_landmarks = []

        if os.path.exists(pts_dir):
            existing_pts = sorted([f for f in os.listdir(pts_dir) if f.endswith(".pts")])
            start_frame = len(existing_pts)
            if start_frame > 0:
                print(f"[RESUME FRAME] Resuming from frame {start_frame}")

                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        all_landmarks = list(pickle.load(f)[:start_frame])
                else:
                    print(f"[RESUME] Re-building {pkl_path} from existing .pts ")
                    for pts_file in existing_pts:
                        pts_path = os.path.join(pts_dir, pts_file)
                        with open(pts_path, "r") as pf:
                            lines = pf.readlines()[1:]
                            pts = [tuple(map(float, line.strip().split())) for line in lines]
                            all_landmarks.append(np.array(pts))
                    save_landmarks_to_pkl(all_landmarks, pkl_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Number of Frames: {total_frames}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < start_frame:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            if len(faces) == 0:
                print(f"[FRAME {frame_idx}] Nessun volto trovato")
                frame_idx += 1
                continue

            shape = landmark_predictor(gray, faces[0])
            points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            save_pts_file(points, os.path.join(pts_dir, f"frame_{frame_idx:03d}.pts"))
            all_landmarks.append(np.array(points))

            if frame_idx % 20 == 0:
                save_landmarks_to_pkl(all_landmarks, pkl_path)

            frame_idx += 1

        cap.release()
        save_landmarks_to_pkl(all_landmarks, pkl_path)

# CREATE ANNOTATIONS FILES
labels = sorted(os.listdir(output_root))
label_to_index = {label: idx for idx, label in enumerate(labels)}

all_samples = []
for label in labels:
    label_dir = os.path.join(output_root, label)
    if not os.path.isdir(label_dir):
        continue

    for fname in os.listdir(label_dir):
        if fname.endswith(".mp4"):
            pkl_path = os.path.join(landmark_root, label, f"{os.path.splitext(fname)[0]}.pkl")
            if os.path.exists(pkl_path):
                speaker = fname.split("_")[0] if "_" in fname else "UNK"
                rel_path = os.path.join(label, fname).replace("\\", "/")
                label_idx = label_to_index[label]
                all_samples.append((rel_path, label_idx, speaker))

write_annotation_file(ANNOTATION_FILE, all_samples)

with open(TEST_CSV, "w") as f_out:
    for rel_path, _, _ in all_samples:
        rel_path = rel_path.replace(".mp4", "")
        f_out.write(f"{rel_path},0\n")
print(f"[CSV] Created {TEST_CSV}")
