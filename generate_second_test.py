#RUN THIS CELL AFTER THE PREPROCESSING AND BEFORE STARTING THE TESTING FACE

import urllib.request
import bz2
import shutil
import os
import cv2
import dlib
import numpy as np
import pickle
from tqdm import tqdm

npz_dir = r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\datasets\visual_data"
output_txt = r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\annotations\test2.txt"

with open(output_txt, "w") as f:
    for label in os.listdir(npz_dir):
        label_dir = os.path.join(npz_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".npz"):
                sample_id = f"{label}/{os.path.splitext(fname)[0]}"
                f.write(sample_id + "\n")

print(f"[ANNOTATIONS] Created file test2.txt with {sum(1 for _ in open(output_txt))} sample")