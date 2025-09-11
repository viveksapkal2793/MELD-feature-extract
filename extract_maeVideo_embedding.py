# @Time    : 6/24/23 9:33 AM
# @Author  : bbbdbbb
# @File    : extract_maeVideo_embedding.py
# @Description : load maeVideo model to extract video feature embedding

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Overwriting.*in registry")
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import shutil
from PIL import Image

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from timm.models import create_model

import sys
sys.path.append('../../')

from maeVideo import models_vit
from collections import OrderedDict
from maeVideo.modeling_finetune import vit_large_patch16_224
from maeVideo.video_transform import *


class SmartVideoDataset(torch.utils.data.Dataset):
    """Dataset that extracts only needed frames from videos on-the-fly"""
    def __init__(self, video_path, num_segments=8, new_length=2, transform=None):
        self.video_path = video_path
        self.num_segments = num_segments
        self.new_length = new_length
        self.transform = transform
        
        # Get total frame count without extracting frames
        self.total_frames = self._get_frame_count()
        
        # Calculate which frames we need
        self.frame_indices = self._sample_indices()
        
        # Extract only the needed frames
        self.frames = self._extract_selected_frames()
        
    def _get_frame_count(self):
        """Get total number of frames in video without extracting them"""
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
    def _sample_indices(self):
        """Sample frame indices using the same logic as original dataset"""
        if self.total_frames > self.num_segments + self.new_length - 1:
            tick = (self.total_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
            
        indices = []
        for seg_ind in range(self.num_segments):
            offset = offsets[seg_ind]
            for i in range(self.new_length):
                frame_index = offset + i
                if frame_index < self.total_frames:
                    indices.append(int(frame_index))
                else:
                    indices.append(int(self.total_frames - 1))
                    
        return sorted(set(indices))  # Remove duplicates and sort
    
    def _extract_selected_frames(self):
        """Extract only the frames we need"""
        self.temp_dir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(self.video_path)
        
        frame_paths = []
        current_frame = 0
        frame_index_set = set(self.frame_indices)
        
        while cap.isOpened() and current_frame <= max(self.frame_indices):
            success, frame = cap.read()
            if not success:
                break
                
            if current_frame in frame_index_set:
                # Save this frame
                frame_filename = f"{current_frame:05d}.bmp"
                frame_path = os.path.join(self.temp_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            current_frame += 1
            
        cap.release()
        
        # Pad to get exactly 16 frames (8 segments * 2 frames each)
        while len(frame_paths) < 16:
            frame_paths.append(frame_paths[-1] if frame_paths else None)
            
        return frame_paths[:16]  # Ensure exactly 16 frames
    
    def __len__(self):
        return 1  # We return one clip per video
    
    def __getitem__(self, index):
        # Load all sampled frames as PIL Images
        images = []
        for frame_path in self.frames:
            if frame_path and os.path.exists(frame_path):
                img = Image.open(frame_path).convert('RGB')
            else:
                # Create a black image if frame doesn't exist
                img = Image.new('RGB', (224, 224), color='black')
            images.append(img)
            
        # Ensure we have exactly 16 frames
        while len(images) < 16:
            images.append(images[-1] if images else Image.new('RGB', (224, 224), color='black'))
        images = images[:16]
            
        # Apply video transforms
        if self.transform is not None:
            transformed_images = self.transform(images)
        else:
            # Fallback transform
            from maeVideo.video_transform import GroupResize, Stack, ToTorchFormatTensor
            transform = transforms.Compose([
                GroupResize(224),
                Stack(),
                ToTorchFormatTensor()
            ])
            transformed_images = transform(images)
        
        # Reshape following the same pattern as dataset_MER.py
        if isinstance(transformed_images, torch.Tensor):
            # The transform should give us (C*T, H, W), reshape to (T, C, H, W) then transpose
            transformed_images = torch.reshape(transformed_images, (16, 3, 224, 224))  # (T, C, H, W)
            transformed_images = transformed_images.transpose(0, 1)  # -> (C, T, H, W)
        
        # Get video name from path
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return transformed_images, video_name
    
    def cleanup(self):
        """Remove temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MELD', help='input dataset')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='maeVideo_ckp199', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='maeVideo_ckp199', help='pth of pretrain MAE model')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=6, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=1, type=int)

    params = parser.parse_args()

    print(f'==> Extracting maeVideo embedding...')
    
    video_dir = params.video_dir
    save_dir = params.save_dir
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # Check device availability
    if params.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(params.device)
    else:
        params.device = 'cpu'
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU")

    # load model
    model = vit_large_patch16_224()

    if True:
        checkpoint_file = os.path.join(
            # "/scratch/data/bikash_rs/vivek/MELD-feature-extract/models_weights", 
            "D:\Acads\BTP\preprocessing_code\models_weights",
            f"{params.pretrain_model}.pth"
        )
        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=params.device, weights_only=False)

        checkpoint_model = None
        for model_key in 'model|module'.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
            
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            new_size = int((num_patches // (16 // model.patch_embed.tubelet_size)) ** 0.5)
            
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, 16 // model.patch_embed.tubelet_size, orig_size, orig_size,
                                                embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, 16 // model.patch_embed.tubelet_size, new_size,
                                                                    new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        load_state_dict(model, checkpoint_model, prefix='')

    model.to(device)
    
    # if torch.cuda.is_available():
    #     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    #     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # else:
    #     print("CUDA not available, using CPU")

    video_transform = transforms.Compose([
        GroupResize(224),
        Stack(), 
        ToTorchFormatTensor()
    ])

    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f'Find total "{len(video_files)}" videos.')

    for i, video_file in enumerate(video_files, 1):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        
        print(f"Processing video '{video_name}' ({i}/{len(video_files)})...")

        # Create smart dataset that only extracts needed frames
        dataset = SmartVideoDataset(video_path, transform=video_transform)
        
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                        
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=params.batch_size,
                num_workers=10,
                drop_last=True,
            )

            for images, video_name_batch in data_loader:
                images = images.to(device)
                
                with torch.no_grad():
                    embedding = model(images)

                print("embedding :", embedding.shape)
                embedding = embedding.cpu().detach().numpy()

                # Clear GPU cache after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # save results
                EMBEDDING_DIM = max(-1, np.shape(embedding)[-1])
                video_name_single = video_name_batch[0]

                csv_file = os.path.join(save_dir, f'{video_name_single}.npy')
                
                if params.feature_level == 'FRAME':
                    embedding = np.array(embedding).squeeze()
                    if len(embedding) == 0:
                        embedding = np.zeros((1, EMBEDDING_DIM))
                    elif len(embedding.shape) == 1:
                        embedding = embedding[np.newaxis, :]
                    np.save(csv_file, embedding)
                elif params.feature_level == 'BLK':
                    embedding = np.array(embedding)
                    if len(embedding) == 0:
                        embedding = np.zeros((257, EMBEDDING_DIM))
                    elif len(embedding.shape) == 3:
                        embedding = np.mean(embedding, axis=0)
                    np.save(csv_file, embedding)
                else:  # UTTERANCE
                    embedding = np.array(embedding).squeeze()
                    if len(embedding) == 0:
                        embedding = np.zeros((EMBEDDING_DIM,))
                    elif len(embedding.shape) == 2:
                        embedding = np.mean(embedding, axis=0)
                    print("csv_file: ", csv_file)
                    print("embedding: ", embedding.shape)
                    np.save(csv_file, embedding)
                    
        finally:
            # Always cleanup temporary files
            dataset.cleanup()
            print(f"Cleaned up temporary files for {video_name}")

# MER2023
# python -u extract_maeVideo_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# EMER
# python -u extract_maeVideo_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# MER2024
# python -u extract_maeVideo_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'
# MER2024_20000
# python -u extract_maeVideo_embedding.py    --dataset='MER2024_20000' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# DFEW
# python -u extract_maeVideo_embedding.py    --dataset='DFEW' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'