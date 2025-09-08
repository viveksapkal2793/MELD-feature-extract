# @Time    : 6/12/23 11:18 AM
# @Author  : bbbdbbb
# @File    : extract_mae_embedding.py
# @Description : extract embedding from pretrain models of mae
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import shutil

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from PIL import Image
import time
import json
import sys

sys.path.append('../../')
# import config
from dataset import FaceDataset

from mae import models_vit
from mae.util.pos_embed import interpolate_pos_embed
from scipy.stats import norm
import scipy.stats as stats


class TempVideoDataset(torch.utils.data.Dataset):
    """Dataset that extracts frames on-the-fly from a video file"""
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.frames = self._extract_frames_to_temp()
        
    def _extract_frames_to_temp(self):
        """Extract all frames to a temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(self.video_path)
        
        frame_paths = []
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Save frame temporarily
            frame_filename = f"{frame_count + 1:05d}.bmp"
            frame_path = os.path.join(self.temp_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            
        cap.release()
        return frame_paths
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        frame_path = self.frames[index]
        img = Image.open(frame_path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(frame_path)[:-4]
        return img, name
    
    def cleanup(self):
        """Remove temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def extract(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        features, timestamps = [], []
        for images, names in data_loader:
            images = images.to(device)
            outputs = model(images)
            embedding = outputs
            print("embedding shape: ", embedding.shape)

            features.append(embedding.cpu().detach().numpy())
            timestamps.extend(names)

            torch.cuda.empty_cache()
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MELD', help='input dataset')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--start_idx', type=int, default=0, help='start index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=-1, help='end index (exclusive), -1 => all')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='mae_checkpoint-340', help='pth name (without .pth)')
    parser.add_argument('--feature_name', type=str, default='mae_340', help='feature name')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--nb_classes', default=7, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--batch_size', default=512, type=int)
    args = parser.parse_args()

    video_dir = args.video_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Compute list of video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    total = len(video_files)
    s = args.start_idx
    e = args.end_idx if args.end_idx > 0 else total

    # load model (use the existing model-loading code but modify checkpoint_file path)
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    checkpoint_file = os.path.join('/kaggle/input/mae-weights', args.pretrain_model + '.pth')
    checkpoint = torch.load(checkpoint_file, map_location=args.device, weights_only=False)
    checkpoint_model = checkpoint['model']
    model.load_state_dict(checkpoint_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)

    device = torch.device(args.device)
    model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load progress if exists
    progress_file = os.path.join(save_dir, 'progress.json')
    processed = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as pf:
            processed = set(json.load(pf).get('processed', []))

    EMBEDDING_DIM = -1
    for idx in range(s, min(e, total)):
        video_file = video_files[idx]
        if video_file in processed:
            print(f"Skipping {video_file} (already processed)")
            continue

        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        print(f"[{idx+1}/{total}] Processing {video_file}")

        dataset = TempVideoDataset(video_path, transform=transform)
        try:
            if len(dataset) == 0:
                embeddings, framenames = [], []
            else:
                loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=False)
                embeddings, framenames = extract(loader, model, device)

            # save results (same logic as original script)
            # ensure save_dir exists
            if len(embeddings) > 0:
                indexes = np.argsort(framenames)
                embeddings = embeddings[indexes]
                framenames = framenames[indexes]
                EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

            out_file = os.path.join(save_dir, f'{video_name}.npy')
            if args.feature_level == 'FRAME':
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((1, EMBEDDING_DIM))
                elif len(embeddings.shape) == 1:
                    embeddings = embeddings[np.newaxis, :]
                np.save(out_file, embeddings)
            else:
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((EMBEDDING_DIM,))
                elif len(embeddings.shape) == 2:
                    embeddings = np.mean(embeddings, axis=0)
                np.save(out_file, embeddings)

            # mark processed & checkpoint
            processed.add(video_file)
            with open(progress_file, 'w') as pf:
                json.dump({'processed': list(processed)}, pf)

        finally:
            dataset.cleanup()

    print("Chunk finished.")


# EMER
# python -u extract_mae_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# MER2023
# python -u extract_mae_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# MER2024
# python -u extract_mae_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'
# MER2024_20000
# python -u extract_mae_embedding.py    --dataset='MER2024_20000' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'

# DFEW
# python -u extract_mae_embedding.py    --dataset='DFEW' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='mae_checkpoint-340' --feature_name='mae_checkpoint-340'