# @Time    : 10/02/25 
# @Author  : 
# @File    : extract_hubert_embedding.py
# @Description : extract audio embedding from videos using pretrained HuBERT model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import argparse
import numpy as np
import tempfile
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
import soundfile as sf

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from transformers import Wav2Vec2FeatureExtractor, HubertModel


class TempAudioDataset(torch.utils.data.Dataset):
    """Dataset that extracts audio on-the-fly from video files"""
    def __init__(self, video_path, feature_extractor, sampling_rate=16000):
        self.video_path = video_path
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate
        self.audio_samples = self._extract_audio_to_temp()
        
    # def _extract_audio_to_temp(self):
    #     """Extract audio from video to temporary file"""
    #     self.temp_dir = tempfile.mkdtemp()
    #     audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
    #     try:
    #         # Extract audio from video
    #         video = VideoFileClip(self.video_path)
    #         audio = video.audio
            
    #         if audio is None:
    #             print(f"Warning: No audio track found in {self.video_path}")
    #             video.close()
    #             return np.array([])
                
    #         # Write audio file with specific parameters and better error handling
    #         try:
    #             audio.write_audiofile(
    #                 audio_path, 
    #                 fps=self.sampling_rate, 
    #                 codec='pcm_s16le', 
    #                 ffmpeg_params=['-ac', '1'],  # mono
    #                 # verbose=False,
    #                 logger=None,
    #                 # temp_audiofile=None,  # Avoid temp file conflicts
    #                 # remove_temp=True
    #             )
    #         except Exception as audio_error:
    #             print(f"Audio extraction failed with moviepy, trying alternative method: {audio_error}")
    #             # Alternative: Use ffmpeg directly
    #             video.close()
    #             return self._extract_with_ffmpeg()
            
    #         # Close video before reading audio
    #         video.close()
            
    #         # Check if audio file was created successfully
    #         if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
    #             print(f"Warning: Audio file not created properly for {self.video_path}")
    #             return self._extract_with_ffmpeg()
                
    #         # Read audio samples
    #         samples, sr = sf.read(audio_path)
            
    #         # Ensure correct sampling rate
    #         if sr != self.sampling_rate:
    #             print(f"Warning: Expected {self.sampling_rate}Hz, got {sr}Hz")
                
    #         return samples
            
    #     except Exception as e:
    #         print(f"Error extracting audio from {self.video_path}: {e}")
    #         return self._extract_with_ffmpeg()

    def _extract_audio_to_temp(self):
        """Extract audio from video using ffmpeg directly"""
        self.temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        
        try:
            import subprocess
            
            # Use ffmpeg directly - most reliable method
            cmd = [
                'ffmpeg',
                '-i', self.video_path,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',  # audio codec
                '-ac', '1',  # mono
                '-ar', str(self.sampling_rate),  # sample rate
                '-y',  # overwrite output
                '-loglevel', 'quiet',  # suppress ffmpeg output
                audio_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=30,  # 30 second timeout
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode != 0:
                print(f"FFmpeg failed for {os.path.basename(self.video_path)}: {result.stderr}")
                return np.array([])
            
            # Check if audio file was created
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"No audio output for {os.path.basename(self.video_path)}")
                return np.array([])
                
            # Read audio samples
            samples, sr = sf.read(audio_path)
            
            if sr != self.sampling_rate:
                print(f"Sample rate mismatch for {os.path.basename(self.video_path)}: {sr} vs {self.sampling_rate}")
                
            return samples
            
        except subprocess.TimeoutExpired:
            print(f"Timeout extracting audio from {os.path.basename(self.video_path)}")
            return np.array([])
        except Exception as e:
            print(f"Error extracting audio from {os.path.basename(self.video_path)}: {e}")
            return np.array([])

    def _extract_with_ffmpeg(self):
        """Fallback method using ffmpeg directly"""
        audio_path = os.path.join(self.temp_dir, "temp_audio_ffmpeg.wav")
        
        try:
            import subprocess
            
            # Use ffmpeg directly to extract audio
            cmd = [
                'ffmpeg',
                '-i', self.video_path,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',  # audio codec
                '-ac', '1',  # mono
                '-ar', str(self.sampling_rate),  # sample rate
                '-y',  # overwrite output
                '-loglevel', 'error',  # only show errors
                audio_path
            ]
            
            # Run ffmpeg with proper error handling
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode != 0:
                print(f"FFmpeg extraction failed for {self.video_path}: {result.stderr}")
                return np.array([])
            
            # Check if audio file was created
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"Warning: No audio extracted with ffmpeg from {self.video_path}")
                return np.array([])
                
            # Read audio samples
            samples, sr = sf.read(audio_path)
            
            if sr != self.sampling_rate:
                print(f"Warning: Expected {self.sampling_rate}Hz, got {sr}Hz")
                
            return samples
            
        except Exception as e:
            print(f"FFmpeg fallback failed for {self.video_path}: {e}")
            return np.array([])
    
    def __len__(self):
        return 1 if len(self.audio_samples) > 0 else 0
    
    def __getitem__(self, index):
        if len(self.audio_samples) == 0:
            # Return empty tensor if no audio
            return torch.zeros(1, 1), "no_audio"
            
        # Process audio with feature extractor
        input_values = self.feature_extractor(
            self.audio_samples, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).input_values
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return input_values.squeeze(0), video_name
    
    def cleanup(self):
        """Remove temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def extract_hubert_features(data_loader, model, device):
    """Extract HuBERT features from audio data"""
    model.eval()
    with torch.no_grad():
        features, video_names = [], []
        
        for input_values, names in data_loader:
            if input_values.numel() == 1:  # Empty audio case
                # Create zero embedding with correct dimension
                embedding = torch.zeros(1, 1024)  # HuBERT large dimension
            else:
                input_values = input_values.to(device)
                
                # Get hidden states from HuBERT
                outputs = model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (B, T, D)
                
                # Use the last layer and average over time dimension
                audio_feature = torch.stack(hidden_states)[[-1]].sum(dim=0)  # (B, T, D)
                audio_feature = audio_feature[0].detach().unsqueeze(0)  # (1, T, D)
                embedding = torch.mean(audio_feature, dim=1, keepdim=True)  # (1, 1, D)
                embedding = embedding.squeeze(1)  # (1, D)
            
            print(f"Audio embedding shape: {embedding.shape}")
            
            features.append(embedding.cpu().detach().numpy())
            if isinstance(names, str):
                video_names.append(names)
            else:
                video_names.extend(names)
                
        if len(features) > 0:
            features = np.row_stack(features)
        else:
            features = np.array([])
            
        return features, np.array(video_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract HuBERT audio embeddings from videos.')
    parser.add_argument('--dataset', type=str, default='MELD', help='input dataset')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', 
                       help='feature level [UTTERANCE only for audio]')
    parser.add_argument('--hubert_model', type=str, default='facebook/hubert-large-ls960-ft', 
                       help='HuBERT model name or path')
    parser.add_argument('--feature_name', type=str, default='hubert_large_english', 
                       help='name for saved features')
    parser.add_argument('--device', default='cuda:0', help='device to use for extraction')
    parser.add_argument('--batch_size', default=8, type=int, 
                       help='batch size (keep small for audio processing)')
    parser.add_argument('--sampling_rate', default=16000, type=int, 
                       help='audio sampling rate')

    params = parser.parse_args()

    print(f'==> Extracting HuBERT audio embedding...')
    
    video_dir = params.video_dir
    save_dir = params.save_dir
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # Setup device
    if params.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(params.device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        params.device = 'cpu'
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU")

    # Load HuBERT model and feature extractor
    print(f"Loading HuBERT model: {params.hubert_model}")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(params.hubert_model)
        model = HubertModel.from_pretrained(params.hubert_model)
        model.to(device)
        print("HuBERT model loaded successfully")
    except Exception as e:
        print(f"Error loading HuBERT model: {e}")
        print("Make sure you have transformers library installed and internet connection")
        exit(1)

    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    EMBEDDING_DIM = 1024  # HuBERT large embedding dimension
    
    print(f'Found total "{len(video_files)}" videos.')
    
    for i, video_file in enumerate(video_files, 1):
        video_name = os.path.splitext(video_file)[0]  # Remove extension
        video_path = os.path.join(video_dir, video_file)
        
        print(f"Processing video '{video_name}' ({i}/{len(video_files)})...")

        # Create temporary dataset for this video
        dataset = TempAudioDataset(video_path, feature_extractor, params.sampling_rate)
        
        try:
            if len(dataset) == 0:
                print(f"Warning: No audio found in video {video_name}")
                embeddings = np.zeros((EMBEDDING_DIM,))
                video_names = [video_name]
            else:
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,  # Process one video at a time for audio
                    num_workers=0,  # Avoid multiprocessing issues with temp files
                    pin_memory=True if torch.cuda.is_available() else False
                )
                embeddings, video_names = extract_hubert_features(data_loader, model, device)

            # Save results
            if len(embeddings) > 0:
                # For utterance level, we expect one embedding per video
                if embeddings.ndim > 1 and embeddings.shape[0] == 1:
                    embeddings = embeddings.squeeze(0)
                elif len(embeddings) == 0:
                    embeddings = np.zeros((EMBEDDING_DIM,))

            save_file = os.path.join(save_dir, f'{video_name}.npy')
            np.save(save_file, embeddings)
            print(f"Saved audio features for {video_name}: shape {embeddings.shape}")
                
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            # Save zero embedding for failed videos
            embeddings = np.zeros((EMBEDDING_DIM,))
            save_file = os.path.join(save_dir, f'{video_name}.npy')
            np.save(save_file, embeddings)
            
        finally:
            # Always cleanup temporary files
            dataset.cleanup()
            print(f"Cleaned up temporary files for {video_name}")

    print("HuBERT audio feature extraction completed!")

# Usage examples:
# python extract_hubert_embedding.py --video_dir /path/to/videos --save_dir /path/to/save --dataset MELD
# python extract_hubert_embedding.py --video_dir /path/to/videos --save_dir /path/to/save --hubert_model facebook/hubert-large-ls960-ft --device cuda:0