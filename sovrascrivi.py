import urllib.request
import bz2
import shutil
import os
import cv2
import dlib
import numpy as np
import pickle
from tqdm import tqdm

dataloader_patch = '''

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import cv2
import pickle
import numpy as np
from transform import *
from utils import *


class AVSRDataLoader(object):

    def __init__(
        self,
        mean_face_path="20words_mean_face.npy",
        crop_width=96,
        crop_height=96,
        start_idx=48,
        stop_idx=68,
        window_margin=12,
        convert_gray=True):

        self._reference = np.load(os.path.join( os.path.dirname(__file__), mean_face_path))
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._start_idx = start_idx
        self._stop_idx = stop_idx
        self._window_margin = window_margin
        self._convert_gray = convert_gray


    def preprocess(self, video_pathname, landmarks_pathname):
            print(f"[DEBUG][preprocess] Carico landmarks da: {landmarks_pathname}")
            
            if isinstance(landmarks_pathname, str):
                # Controlla se il file è vuoto
                if not os.path.exists(landmarks_pathname) or os.path.getsize(landmarks_pathname) == 0:
                    print(f"[ERROR] Il file dei landmarks non esiste o è vuoto: {landmarks_pathname}")
                    return None
                
                try:
                    with open(landmarks_pathname, "rb") as pf:
                        landmarks = pickle.load(pf)
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"[ERROR] Errore nel caricare il file pickle {landmarks_pathname}: {e}")
                    return None
            else:
                landmarks = landmarks_pathname

            num_none = sum(1 for l in landmarks if l is None)
            print(f"[DEBUG][preprocess] Landmarks caricati. Num frame: {len(landmarks)} | con None: {num_none}")

            preprocessed_landmarks = self.landmarks_interpolate(landmarks)
            none_after = sum(1 for l in preprocessed_landmarks if l is None) if preprocessed_landmarks is not None else 'N/A'
            print(f"[DEBUG] Dopo interpolazione: {len(preprocessed_landmarks)} landmarks | con None: {none_after}")

            cap = cv2.VideoCapture(video_pathname)
            if not cap.isOpened():
                print(f"[ERROR] Impossibile aprire il file video: {video_pathname}")
                return None
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"[DEBUG] Frame video: {video_frames} | Frame landmarks: {len(preprocessed_landmarks)}")
            
            # Aggiunto un controllo per gestire il caso in cui l'interpolazione restituisce None
            if preprocessed_landmarks is None:
                print(f"[ERROR] Interpolazione dei landmarks fallita per {video_pathname}")
                return None

            if video_frames != len(preprocessed_landmarks):
                print(f"[ERROR][preprocess] Mismatch frame-video ({video_frames}) vs landmarks ({len(preprocessed_landmarks)})")
                return None

            n = len(preprocessed_landmarks)
            if n < self._window_margin:
                print(f"[WARNING] Solo {n} frame (< window_margin={self._window_margin}); procedo comunque.")

            sequence, _, _ = self.crop_patch(video_pathname, preprocessed_landmarks)

            if sequence is None or len(sequence) == 0:
                print(f"[ERROR] sequence None o vuota per {video_pathname}")
            else:
                print(f"[DEBUG] sequence generata con {len(sequence)} frame")

            return sequence


    def landmarks_interpolate(self, landmarks):
        print(f"[DEBUG][interpolate] Landmarks in input: {len(landmarks)}")
        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                continue
            else:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

        # -- Corner case: riempi inizio/fine con l’ultimo valido
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

        valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]

        assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"

        print(f"[DEBUG][interpolate] Landmarks dopo interpolazione: {len(landmarks)}")
        return landmarks


    def crop_patch(self, video_pathname, landmarks):

        frame_idx = 0
        frame_gen = load_video(video_pathname)
        while True:
            if frame_idx >= len(landmarks):
              print(f"[ERROR] frame_idx {frame_idx} >= len(landmarks) {len(landmarks)} → desync video/landmark")
              break
            try:
                frame = frame_gen.__next__() ## -- BGR
            except StopIteration:
                break
            if frame_idx == 0:
                sequence = []

                sequence_frame = []
                sequence_landmarks = []
            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            if window_margin < 0:
                window_margin = 0
            smoothed_landmarks = np.mean([landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(frame, smoothed_landmarks, self._reference, grayscale=self._convert_gray)
            sequence.append( cut_patch( transformed_frame,
                                        transformed_landmarks[self._start_idx:self._stop_idx],
                                        self._crop_height//2,
                                        self._crop_width//2,))

            sequence_frame.append( transformed_frame)
            sequence_landmarks.append( transformed_landmarks)

            frame_idx += 1
        return np.array(sequence), np.array(sequence_frame), np.array(sequence_landmarks)


    def affine_transform(self, frame, landmarks, reference, grayscale=False, target_size=(256, 256),
                         reference_size=(256, 256), stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                         border_value=0):
        # Prepare everything
        if grayscale and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable_reference = np.vstack([reference[x] for x in stable_points])
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

        # Warp the face patch and the landmarks
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                                stable_reference, method=cv2.LMEDS)[0]
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                    flags=interpolation, borderMode=border_mode, borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

        return transformed_frame, transformed_landmarks


    def load_audio(self, data_filename):
        sequence = load_audio(data_filename, specified_sr=16000, int_16=False)
        return sequence

    def load_video(self, data_filename, landmarks_filename=None):

        assert landmarks_filename is not None
        sequence = self.preprocess(
            video_pathname=data_filename,
            landmarks_pathname=landmarks_filename,
        )
        return sequence

    def load_data(self, modality, data_filename, landmarks_filename=None):
        if modality == "raw_audio":
            return self.load_audio(data_filename)
        elif modality == "video":
            return self.load_video(data_filename, landmarks_filename=landmarks_filename)
'''

with open(r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\preprocessing\dataloader.py", "w", encoding='utf-8') as f:
    f.write(dataloader_patch)

print("dataloader.py patchato con successo.")


dataloaders_patch = """
import torch
import numpy as np
from lipreading.preprocess import *
from lipreading.dataset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std),
                                    TimeMask(T=0.6*25, n_mask=1)
                                    ])

        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                labels=args.labels,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz',
                use_boundary=args.use_boundary,
                ) for partition in partitions}
    for partition in partitions:
        print(f"[DEBUG] Partition '{partition}': {len(dsets[partition])} samples")
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders

"""

with open(r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\lipreading\dataloaders.py", "w", encoding='utf-8') as f:
    f.write(dataloaders_patch)

print("dataloaders.py patchato con successo.")

dataset_patch = '''

import os
import glob
import torch
import random
import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines


class MyDataset(object):
    def __init__(self, data_partition, modality, data_dir, label_fp, labels, annonation_direc=None,
                 preprocessing_func=None, data_suffix='.npz', use_boundary=False):

        assert os.path.isfile(label_fp), \
            f"File path provided for the labels does not exist. Path input: {label_fp}."

        self.data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix
        self._label_fp = label_fp
        self.labels = labels
        print("[CONTROL]: ", labels)
        self._annonation_direc = annonation_direc

        self.fps = 25 if modality == "video" else 16000
        self.is_var_length = False
        self.use_boundary = use_boundary
        self.label_idx = -2
        self.preprocessing_func = preprocessing_func

        if self.use_boundary or (self.is_var_length):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        print(f"[DEBUG][{modality}] Inizializzazione dataset")
        print(f"[DEBUG] data_dir = {self._data_dir}")
        print(f"[DEBUG] label_fp = {self._label_fp}")

        self.data = []

        # Aggiunta solo di log di debug: stampiamo cosa sta cercando di caricare
        with open(label_fp, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            rel_path = parts[0]  # solo il path relativo al file .mp4
            npz_path = os.path.join(self._data_dir, rel_path + self._data_suffix)
            if os.path.exists(npz_path):
                print(f"[FOUND] {npz_path}")
                self.data.append(npz_path)
            else:
                print(f"[MISSING] {npz_path}")

        print(f"[DEBUG] Trovati {len(self.data)} esempi validi per il partition")

        self._data_files = []
        self.load_dataset()

    def load_dataset(self):

        # -- read the labels file
        self._labels = read_txt_lines(self.labels)
        print("[DEBUG labels]", self._labels)


        # -- add examples to self._data_files
        self._get_files_for_partition()


        # -- from self._data_files to self.list
        self.list = dict()
        self.instance_ids = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path( x )
            print(f"[DEBUG] path: {x} → label estratta: {label}")
            self.list[i] = [ x, self._labels.index( label ) ]
            self.instance_ids[i] = self._get_instance_id_from_path( x )

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        # --- INIZIO CORREZIONE 1/3
        instance_id = os.path.basename(x)
        return os.path.splitext( instance_id )[0]
        # --- FINE CORREZIONE 1/3

    def _get_label_from_path(self, x):
        # --- INIZIO CORREZIONE 2/3
        return os.path.basename(os.path.dirname(x))
        # --- FINE CORREZIONE 2/3

    def _get_files_for_partition(self):
        # Cerca i file npz/npy/mp4 in tutte le sottocartelle (label)
        dir_fp = self._data_dir
        if not dir_fp:
            print("[ERRORE] data_dir è None o vuota!")
            return

        # Utilizza glob.glob che è già compatibile con entrambi i separatori
        search_str_npz = os.path.join(dir_fp, '*', '*.npz')
        search_str_npy = os.path.join(dir_fp, '*', '*.npy')
        search_str_mp4 = os.path.join(dir_fp, '*', '*.mp4')

        self._data_files.extend(glob.glob(search_str_npz))
        self._data_files.extend(glob.glob(search_str_npy))
        self._data_files.extend(glob.glob(search_str_mp4))

        print(f"[DEBUG] Trovati {len(self._data_files)} file prima del filtraggio")
        print("[DEBUG] Prime 5 path trovate:")
        for f in self._data_files[:5]:
            print(f"  - {f}")

        # Mostra le label attese caricate dal file
        print(f"[DEBUG] Numero totale di label attese: {len(self._labels)}")
        print("[DEBUG] Esempi di label attese:")
        for lbl in self._labels[:10]:
            print(f"  - '{lbl}'")

        match_count = 0
        mismatch_count = 0

        # Analizza tutti i file e diagnostica il motivo del filtraggio
        for f in self._data_files:
            # Usa os.path.normpath per uniformare i separatori e os.path.split per estrarre il nome della directory
            parent_dir = os.path.normpath(f).split(os.path.sep)[-2]
            label_match = parent_dir in self._labels

            if label_match:
                match_count += 1
            else:
                mismatch_count += 1
                print(f"[MISMATCH] Directory '{parent_dir}' NON trovata tra le label attese → {f}")

        # Effettua il filtraggio
        self._data_files = [
            f for f in self._data_files
            if os.path.normpath(f).split(os.path.sep)[-2] in self._labels
        ]

        print(f"[DEBUG] Trovati {len(self._data_files)} file dopo il filtraggio per label")
        print(f"[DEBUG] Totale match: {match_count}, mismatch: {mismatch_count}")

        # Esempi finali dei file mantenuti
        print("[DEBUG] Esempi di file accettati dopo filtraggio:")
        for f in self._data_files[:5]:
            print(f"  - {f}")

        # Diagnostica avanzata: se 0 file trovati, controlla case sensitivity
        if match_count == 0:
            print("[POSSIBILE PROBLEMA] Nessuna directory trovata tra le label!")
            print("Verifica se le label hanno problemi di maiuscole/minuscole.")
            print("Esempio confronto tra directory e label attese:")
            example_dir_names = set([os.path.normpath(f).split(os.path.sep)[-2] for f in self._data_files])
            for example in list(example_dir_names)[:5]:
                in_labels = example in self._labels
                in_labels_lower = example.lower() in [l.lower() for l in self._labels]
                print(f"  '{example}' in labels? → {in_labels}")
                print(f"  '{example.lower()}' in labels.lower()? → {in_labels_lower}")


    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        # --- INIZIO CORREZIONE 3/3
        # L'uso di `*filename.split('/')` non è compatibile con Windows
        rel_path_parts = filename.split(os.path.sep)[self.label_idx:]
        info_txt = os.path.join(self._annonation_direc, *rel_path_parts)
        # --- FINE CORREZIONE 3/3
        
        info_txt = os.path.splitext( info_txt )[0] + '.txt'  # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        half_interval = int( utterance_duration/2.0 * self.fps)  # num frames of utterance / 2

        n_frames = raw_data.shape[0]
        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0,mid_idx-half_interval-1)  )  # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint( min( mid_idx+half_interval+1,n_frames ), n_frames  )

        return raw_data[left_idx:right_idx]


    def _get_boundary(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        # --- INIZIO CORREZIONE 3/3
        # L'uso di `*filename.split('/')` non è compatibile con Windows
        rel_path_parts = filename.split(os.path.sep)[self.label_idx:]
        info_txt = os.path.join(self._annonation_direc, *rel_path_parts)
        # --- FINE CORREZIONE 3/3
        
        info_txt = os.path.splitext( info_txt )[0] + '.txt'  # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float( info[4].split(' ')[1] )
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int( utterance_duration/2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int( utterance_duration/2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = ( n_frames -1 ) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx-half_interval-1)
        right_idx = min(mid_idx+half_interval+1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def load_data(self, path):
        arr = np.load(path)
        return arr['data'] if 'data' in arr else arr

    def __getitem__(self, idx):
        path, label = self.list[idx]
        print(f"[__getitem__] Loading sample {idx} | Path: {path} | Label: {label}")

        try:
            raw = self.load_data(path)
            if raw.size == 0:
                raise ValueError("Loaded data is empty.")
        except Exception as e:
            # Gestisce il caso di file .npz vuoti o corrotti
            print(f"[ERROR] Impossibile caricare il file {path}: {e}")
            return None, None
        
        # Se è RGB, converti in scala di grigio
        if raw.ndim == 4 and raw.shape[-1] == 3:
            r, g, b = raw[..., 0], raw[..., 1], raw[..., 2]
            raw = 0.2989*r + 0.5870*g + 0.1140*b  # (T, H, W)
            print(f"[__getitem__] Converted to grayscale: shape = {raw.shape}")

        if self.is_var_length and not self.use_boundary:
            raw = self._apply_variable_length_aug(path, raw)
            print(f"[__getitem__] Applied variable length augmentation.")

        data = self.preprocessing_func(raw) if self.preprocessing_func else raw
        print(f"[__getitem__] Preprocessed data shape: {data.shape}")

        if self.use_boundary:
            boundary = self._get_boundary(path, raw)
            print(f"[__getitem__] Boundary shape: {boundary.shape}")
            return data, label, boundary

        return data, label


    def __len__(self):
        return len(self._data_files)


def pad_packed_collate(batch):
    # Filtra i campioni che sono stati saltati (__getitem__ ha restituito None)
    batch = [item for item in batch if item[0] is not None]

    if not batch:
        # Restituisci tensori vuoti per evitare errori se l'intero batch è corrotto
        return torch.FloatTensor([]), [], torch.LongTensor([]), None
    
    if len(batch[0]) == 2:
        use_boundary = False
        data_tuple, lengths, labels_tuple = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
    elif len(batch[0]) == 3:
        use_boundary = True
        data_tuple, lengths, labels_tuple, boundaries_tuple = zip(*[(a, a.shape[0], b, c) for (a, b, c) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

    if data_tuple[0].ndim == 1:
        max_len = data_tuple[0].shape[0]
        data_np = np.zeros((len(data_tuple), max_len))
    elif data_tuple[0].ndim == 3:
        max_len, h, w = data_tuple[0].shape
        data_np = np.zeros((len(data_tuple), max_len, h, w))
    for idx in range( len(data_np)):
        data_np[idx][:data_tuple[idx].shape[0]] = data_tuple[idx]
    data = torch.FloatTensor(data_np)

    if use_boundary:
        boundaries_np = np.zeros((len(boundaries_tuple), len(boundaries_tuple[0])))
        for idx in range(len(data_np)):
            boundaries_np[idx] = boundaries_tuple[idx]
        boundaries = torch.FloatTensor(boundaries_np).unsqueeze(-1)

    labels = torch.LongTensor(labels_tuple)

    if use_boundary:
        return data, lengths, labels, boundaries
    else:
        return data, lengths, labels

'''

with open(r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\lipreading\dataset.py", "w", encoding='utf-8') as f:
    f.write(dataset_patch)

print("Dataset sovrascritto con successo.")


new_main = '''

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'audio'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with paths to visual_data')
    parser.add_argument('--labels', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- DenseTCN config
    parser.add_argument('--densetcn-block-config', type=int, nargs = "+", help='number of denselayer for each denseTCN block')
    parser.add_argument('--densetcn-kernel-size-set', type=int, nargs = "+", help='kernel size set for each denseTCN block')
    parser.add_argument('--densetcn-dilation-size-set', type=int, nargs = "+", help='dilation size set for each denseTCN block')
    parser.add_argument('--densetcn-growth-rate-set', type=int, nargs = "+", help='growth rate for DenseTCN')
    parser.add_argument('--densetcn-dropout', default=0.2, type=float, help='Dropout value for DenseTCN')
    parser.add_argument('--densetcn-reduced-size', default=256, type=int, help='the feature dim for the output of reduce layer')
    parser.add_argument('--densetcn-se', default = False, action='store_true', help='If True, enable SE in DenseTCN')
    parser.add_argument('--densetcn-condense', default = False, action='store_true', help='If True, enable condenseTCN')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                         help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')
    # use boundaries
    parser.add_argument('--use-boundary', default=False, action='store_true', help='include hard border at the testing stage.')

    args = parser.parse_args()
    return args


args = load_args()

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True

# Seleziona il dispositivo (CPU)
device = torch.device("cpu")

def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].to(device), lengths=[data.shape[0]])


def evaluate(model, dset_loader, criterion, idx_to_word=None):
    model.eval()
    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader)):
            if args.use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.to(device)
            else:
                input, lengths, labels = data
                boundaries = None

            input = input.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            logits = model(input, lengths=lengths, boundaries=boundaries)
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)

            running_corrects += preds.eq(labels.view_as(preds)).sum().item()
            loss = criterion(logits, labels)
            running_loss += loss.item() * input.size(0)

            if idx_to_word:
                for i in range(len(labels)):
                    true_idx = labels[i].item()
                    pred_idx = preds[i].item()

                    true_word = idx_to_word.get(true_idx)
                    pred_word = idx_to_word.get(pred_idx)

                    if true_word is None:
                        print(f"[WARN] Missing GT label: {true_idx}")
                    if pred_word is None:
                        print(f"[WARN] Missing PRED label: {pred_idx}")

                    true_word = true_word if true_word else f"<UNK:{true_idx}>"
                    pred_word = pred_word if pred_word else f"<UNK:{pred_idx}>"
                    print(f"[EVAL] Sample {i:04d} | GT: {true_word}  | Pred: {pred_word}  {'✅' if true_idx == pred_idx else '❌'}")

    total = len(dset_loader.dataset)
    acc = running_corrects / total
    avg_loss = running_loss / total
    print(f"[EVAL] {total} samples total\tAccuracy: {acc:.4f}\tAvg Loss: {avg_loss:.4f}")
    return acc, avg_loss


def train(model, dset_loader, criterion, epoch, optimizer, logger):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info(f"Epoch {epoch}/{args.epochs - 1}")
    logger.info(f"Current learning rate: {lr}")

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.

    end = time.time()
    for batch_idx, data in enumerate(dset_loader):
        if args.use_boundary:
            input, lengths, labels, boundaries = data
            boundaries = boundaries.to(device)
        else:
            input, lengths, labels = data
            boundaries = None
        data_time.update(time.time() - end)

        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.to(device), labels_b.to(device)

        optimizer.zero_grad()

        logits = model(input.unsqueeze(1).to(device), lengths=lengths, boundaries=boundaries)
        
        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
        running_loss += loss.item()*input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        if batch_idx % args.interval == 0 or (batch_idx == len(dset_loader)-1):
            update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    return model


def get_model_from_json():
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"'.json' config path does not exist. Path input: {args.config_path}"
    args_loaded = load_json( args.config_path)
    args.backbone_type = args_loaded['backbone_type']
    args.width_mult = args_loaded['width_mult']
    args.relu_type = args_loaded['relu_type']
    args.use_boundary = args_loaded.get("use_boundary", False)

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    model = Lipreading( modality=args.modality,
                        num_classes=args.num_classes,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        backbone_type=args.backbone_type,
                        relu_type=args.relu_type,
                        width_mult=args.width_mult,
                        use_boundary=args.use_boundary,
                        extract_feats=args.extract_feats)
    
    calculateNorm2(model)
    return model


def get_label_mappings(npz_dir):
    labels = sorted([d for d in os.listdir(npz_dir) if os.path.isdir(os.path.join(npz_dir, d))])
    idx_to_word = {i: label for i, label in enumerate(labels)}
    word_to_idx = {label: i for i, label in enumerate(labels)}
    return idx_to_word, word_to_idx

def get_label_mappings_from_file(label_file_path):
    idx_to_word = {}
    word_to_idx = {}
    with open(label_file_path, 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            if word:  # Skip blank lines
                idx_to_word[idx] = word
                word_to_idx[word] = idx
    return idx_to_word, word_to_idx

def main():

    # -- logging
    save_path = get_save_folder(args)
    print(f"Model and log being saved in: {save_path}")
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    # -- get model
    model = get_model_from_json()

    # Sposta il modello sulla CPU qui, dopo l'inizializzazione
    model = model.to(device)

    # -- get dataset iterators
    dset_loaders = get_data_loaders(args)

    # -- get loss function
    criterion = nn.CrossEntropyLoss()

    # -- get optimizer
    optimizer = get_optimizer(args, optim_policies=model.parameters())

    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)

    # -- get label mappings
    label_file = "C:/Users/adria/OneDrive/Documenti/Università/NLP/project/LRS2_Test/Lipreading_using_Temporal_Convolutional_Networks/labels/500WordsSortedList.txt"
    idx_to_word, word_to_idx = get_label_mappings_from_file(label_file)
    print(f"[DEBUG] Loaded {len(idx_to_word)} labels in idx_to_word")

    # -- model resume or load
    if args.model_path:
        assert os.path.isfile(args.model_path), f"Model path does not exist: {args.model_path}"
        
        # Gestione del caricamento del modello
        if args.init_epoch > 0:
            # Caricamento di un checkpoint completo (per riprendere il training)
            model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
            args.init_epoch = epoch_idx
            ckpt_saver.set_best_from_ckpt(ckpt_dict)
            logger.info(f"Model and states loaded from {args.model_path}")
        else:
            # Caricamento solo dei pesi del modello (per il testing)
            model, _, _, _ = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
            logger.info(f"Model loaded from {args.model_path}")
            
        # Sposta il modello sul dispositivo corretto dopo il caricamento
        model = model.to(device)

        if args.mouth_patch_path:
            save2npz(args.mouth_embedding_out_path, data=extract_feats(model).cpu().detach().numpy())
            return

        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, idx_to_word=idx_to_word)
            logger.info(f"[TEST] Loss: {loss_avg_test:.4f}\tAccuracy: {acc_avg_test:.4f}")
            return

    # -- fix learning rate after checkpoint resume
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch - 1)

    epoch = args.init_epoch

    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion, idx_to_word=idx_to_word)
        logger.info(f"[VAL] Epoch {epoch} | Loss: {loss_avg_val:.4f} | Accuracy: {acc_avg_val:.4f} | LR: {showLR(optimizer)}")

        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- final evaluation
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    # Carica il modello migliore salvato per l'ultima valutazione
    _ = load_model(best_fp, model)
    model = model.to(device)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, idx_to_word=idx_to_word)
    logger.info(f"[BEST TEST] Loss: {loss_avg_test:.4f}\tAccuracy: {acc_avg_test:.4f}")



if __name__ == '__main__':
    main()
'''

with open(r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\main.py", "w", encoding='utf-8') as f:
    f.write(new_main)

print("Main sovrascritto con successo.")


utils = '''
import os
import json
import numpy as np

import datetime
import logging

import json
import torch
import shutil


def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    print('2-norm of the neural network: {:.4f}'.format(para_norm**.5))


def showLR(optimizer):
    return optimizer.param_groups[0]['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -- IO utils
def read_txt_lines(filepath):
    assert os.path.isfile( filepath ), "Error when trying to read txt file, path does not exist: {}".format(filepath)
    with open( filepath ) as myfile:
        content = myfile.read().splitlines()
    return content


def save_as_json(d, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def load_json( json_fp ):
    assert os.path.isfile( json_fp ), "Error loading JSON. File provided does not exist, cannot read: {}".format( json_fp )
    with open( json_fp, 'r' ) as f:
        json_content = json.load(f)
    return json_content


def save2npz(filename, data=None):
    assert data is not None, "data is {}".format(data)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.savez_compressed(filename, data=data)


# -- checkpoints
class CheckpointSaver:
    def __init__(self, save_dir, checkpoint_fn='ckpt.pth', best_fn='ckpt.best.pth', best_step_fn='ckpt.best.step{}.pth', save_best_step=False, lr_steps=[]):
        """
        Only mandatory: save_dir
            Can configure naming of checkpoint files through checkpoint_fn, best_fn and best_stage_fn
            If you want to keep best-performing checkpoint per step
        """

        self.save_dir = save_dir

        # checkpoint names
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn

        # save best per step?
        self.save_best_step = save_best_step
        self.lr_steps = []

        # init var to keep track of best performing checkpoint
        self.current_best = 0

        # save best at each step?
        if self.save_best_step:
            assert lr_steps != [], "Since save_best_step=True, need proper value for lr_steps. Current: {}".format(lr_steps)
            self.best_for_stage = [0]*(len(lr_steps)+1)

    def save(self, save_dict, current_perf, epoch=-1):
        """
            Save checkpoint and keeps copy if current perf is best overall or [optional] best for current LR step
        """

        # save last checkpoint
        checkpoint_fp = os.path.join(self.save_dir, self.checkpoint_fn)

        # keep track of best model
        self.is_best = current_perf > self.current_best
        if self.is_best:
            self.current_best = current_perf
            best_fp = os.path.join(self.save_dir, self.best_fn)
        save_dict['best_prec'] = self.current_best

        # keep track of best-performing model per step [optional]
        if self.save_best_step:

            assert epoch >= 0, "Since save_best_step=True, need proper value for 'epoch'. Current: {}".format(epoch)
            s_idx = sum( epoch >= l for l in lr_steps )
            self.is_best_for_stage = current_perf > self.best_for_stage[s_idx]

            if self.is_best_for_stage:
                self.best_for_stage[s_idx] = current_perf
                best_stage_fp = os.path.join(self.save_dir, self.best_stage_fn.format(s_idx))
            save_dict['best_prec_per_stage'] = self.best_for_stage

        # save
        torch.save(save_dict, checkpoint_fp)
        print("Checkpoint saved at {}".format(checkpoint_fp))
        if self.is_best:
            shutil.copyfile(checkpoint_fp, best_fp)
        if self.save_best_step and self.is_best_for_stage:
            shutil.copyfile(checkpoint_fp, best_stage_fp)


    def set_best_from_ckpt(self, ckpt_dict):
        self.current_best = ckpt_dict['best_prec']
        self.best_for_stage = ckpt_dict.get('best_prec_per_stage',None)


def load_model(load_path, model, optimizer=None, device='cpu', allow_size_mismatch=False):
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"File does not exist: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict):
        # This handles both simple state dicts and full checkpoints
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=not allow_size_mismatch)
        
        # Only load optimizer and epoch if they exist
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Return different values based on whether the full checkpoint is available
        if 'epoch_idx' in checkpoint:
            return model, optimizer, checkpoint['epoch_idx'], checkpoint
        else:
            # Handle cases where the checkpoint is just a model state dict
            return model, optimizer, -1, None
            
    else:
        # If the checkpoint is a simple model state dict without a dictionary wrapper
        model.load_state_dict(checkpoint, strict=not allow_size_mismatch)
        return model, optimizer, -1, None


# -- logging utils
def get_logger(args,save_path):
    log_path = '{}/{}_{}_{}classes_log.txt'.format(save_path,args.training_mode,args.lr,args.num_classes)
    logger = logging.getLogger("mylog")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


def update_logger_batch( args, logger, dset_loader, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time ):
    perc_epoch = 100. * batch_idx / (len(dset_loader)-1)
    logger.info(f"[{running_all:5.0f}/{len(dset_loader.dataset):5.0f} ({perc_epoch:.0f}%)]\tLoss: {running_loss / running_all:.4f}\tAcc:{running_corrects / running_all:.4f}\tCost time:{batch_time.val:1.3f} ({batch_time.avg:1.3f})s\tData time:{data_time.val:1.3f} ({data_time.avg:1.3f})\tInstances per second: {args.batch_size/batch_time.avg:.2f}")

def get_save_folder( args):
    # create save and log folder
    save_path = '{}/{}'.format( args.logging_dir, args.training_mode )
    
    # Generate a timestamp and replace colons with hyphens for Windows compatibility
    timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '-')
    save_path += '/' + timestamp
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    return save_path

'''

with open(r"Path_To_Project\Lipreading_using_Temporal_Convolutional_Networks\lipreading\utils.py", "w", encoding='utf-8') as f:
    f.write(utils)

print("Utils sovrascritto con successo.")