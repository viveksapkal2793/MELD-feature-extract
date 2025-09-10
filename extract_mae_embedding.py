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
        features, timestamps = np.row_stack(features), np.array(timestamps)
        return features, timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MELD', help='input dataset')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='mae_checkpoint-340', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='mae_340', help='pth of pretrain MAE model')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train vit_large_patch16 vit_huge_patch14')
    parser.add_argument('--nb_classes', default=7, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=512, type=int)

    params = parser.parse_args()

    print(f'==> Extracting mae embedding...')
    
    video_dir = params.video_dir
    save_dir = params.save_dir
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # load model
    model = models_vit.__dict__[params.model](
        num_classes=params.nb_classes,
        drop_path_rate=params.drop_path,
        global_pool=params.global_pool,
    )

    if params.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(params.device)
    else:
        params.device = 'cpu'
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU")
    
    if True:
        checkpoint_file = os.path.join(
            "/scratch/data/bikash_rs/vivek/MELD-feature-extract/models_weights", 
            f"{params.pretrain_model}.pth"
        )
        checkpoint = torch.load(checkpoint_file, map_location=params.device, weights_only=False)

        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint_model = checkpoint['model']

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)
        trunc_normal_(model.head.weight, std=2e-5)

    device = torch.device(params.device)
    model.to(device)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    EMBEDDING_DIM = -1
    print(f'Find total "{len(video_files)}" videos.')
    
    for i, video_file in enumerate(video_files, 1):
        video_name = os.path.splitext(video_file)[0]  # Remove .mp4 extension
        video_path = os.path.join(video_dir, video_file)
        
        print(f"Processing video '{video_name}' ({i}/{len(video_files)})...")

        # Create temporary dataset
        dataset = TempVideoDataset(video_path, transform=transform)
        
        try:
            if len(dataset) == 0:
                print("Warning: number of frames of video {} should not be zero.".format(video_name))
                embeddings, framenames = [], []
            else:
                data_loader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=params.batch_size,
                                                          num_workers=4,  # Reduced workers for temp files
                                                          pin_memory=True)
                embeddings, framenames = extract(data_loader, model, device)

            # save results
            if len(embeddings) > 0:
                indexes = np.argsort(framenames)
                embeddings = embeddings[indexes]
                framenames = framenames[indexes]
                EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

            csv_file = os.path.join(save_dir, f'{video_name}.npy')
            if params.feature_level == 'FRAME':
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((1, EMBEDDING_DIM))
                elif len(embeddings.shape) == 1:
                    embeddings = embeddings[np.newaxis, :]
                np.save(csv_file, embeddings)
            else:  # UTTERANCE level
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((EMBEDDING_DIM,))
                elif len(embeddings.shape) == 2:
                    embeddings = np.mean(embeddings, axis=0)
                np.save(csv_file, embeddings)
                
        finally:
            # Always cleanup temporary files
            dataset.cleanup()
            print(f"Cleaned up temporary files for {video_name}")


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