#!/usr/bin/env python3
"""
Clean EVA-ViT feature extraction script with organized dependencies.
Extracts 1408-dimensional features from peak frame images.
Can extract peak frames on-the-fly or use pre-extracted frames.
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import logging
import tempfile
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dependencies import create_eva_vit_g, BlipImageEvalProcessor
from peak_frame_ext import PeakFrameExtractor

class EVAViTFeatureExtractor:
    def __init__(self, device='cuda', precision='fp16', weights_dir='./weights'):
        """
        Initialize EVA-ViT model for feature extraction.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            precision: Model precision ('fp16' or 'fp32')
            weights_dir: Directory to download/store model weights
        """
        self.device = device
        self.precision = precision
        self.weights_dir = weights_dir
        
        # Initialize EVA-ViT model
        print("Loading EVA-ViT model...")
        self.model = create_eva_vit_g(
            img_size=448,
            drop_path_rate=0.0,
            use_checkpoint=False,
            precision=precision,
            weights_dir=weights_dir
        )
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize image processor
        self.vis_processor = BlipImageEvalProcessor(image_size=448)
        
        print(f"EVA-ViT model loaded on {device} with {precision} precision")
        print(f"Model feature dimension: {self.model.num_features}")
        print(f"Output shape will be: [1025, 1408] (CLS + patches, features)")
        
        # Initialize peak frame extractor for on-the-fly extraction
        self.peak_extractor = PeakFrameExtractor(use_opencv=True)
    
    def extract_features(self, image):
        """
        Extract EVA-ViT features from image.
        
        Args:
            image: PIL Image
            
        Returns:
            numpy array of shape [1025, 1408] (CLS + patch tokens, feature_dim)
        """
        try:
            # Preprocess image
            image_tensor = self.vis_processor(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if self.precision == 'fp16' and self.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        features = self.model(image_tensor)  # [1, 1025, 1408]
                else:
                    features = self.model(image_tensor)  # [1, 1025, 1408]
            
            # Convert to numpy and remove batch dimension
            features_np = features.squeeze(0).cpu().numpy()  # [1025, 1408]
            
            return features_np
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None
    
    def process_peak_frames_from_directory(self, peak_frames_dir, output_dir, image_extensions=None):
        """
        Process pre-extracted peak frame images from directory.
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(peak_frames_dir, f"*{ext}")
            found_files = glob.glob(pattern)
            image_files.extend(found_files)
            pattern = os.path.join(peak_frames_dir, f"*{ext.upper()}")
            found_files_upper = glob.glob(pattern)
            image_files.extend(found_files_upper)
        
        # Remove duplicates (in case same file is found with different case)
        image_files = list(set(image_files))
        
        print(f"Found {len(image_files)} peak frame images")
        if len(image_files) == 0:
            print(f"No images found in {peak_frames_dir}")
            print(f"Looking for extensions: {image_extensions}")
            return
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_files, desc="Processing peak frames"):
            try:
                # Get image name without extension
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Check if features already exist
                output_path = os.path.join(output_dir, f"{image_name}.npy")
                if os.path.exists(output_path):
                    logging.info(f"Features already exist for {image_name}, skipping...")
                    successful += 1  # Count as successful
                    continue
                
                # Load peak frame
                try:
                    peak_frame = Image.open(image_path).convert('RGB')
                except Exception as e:
                    logging.error(f"Failed to load image {image_path}: {str(e)}")
                    failed += 1
                    continue
                
                # Extract features
                features = self.extract_features(peak_frame)
                if features is None:
                    failed += 1
                    continue
                
                # Save features
                np.save(output_path, features)
                successful += 1
                
                if successful % 100 == 0:  # Log every 100 images
                    logging.info(f"Processed {successful} images, current: {image_name} -> {features.shape}")
                
            except Exception as e:
                logging.error(f"Failed to process {image_path}: {str(e)}")
                failed += 1
        
        return successful, failed
    
    def process_peak_frames_on_the_fly(self, video_dir, csv_dir, output_dir, temp_dir=None, cleanup=True):
        """
        Extract peak frames on-the-fly and process them for feature extraction.
        
        Args:
            video_dir: Directory containing video files
            csv_dir: Directory containing OpenFace CSV files
            output_dir: Directory to save extracted features
            temp_dir: Temporary directory for peak frames (created if None)
            cleanup: Whether to delete temporary peak frames after processing
        """
        video_dir = Path(video_dir)
        csv_dir = Path(csv_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create or use provided temp directory
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="peak_frames_"))
            temp_created = True
        else:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_created = False
        
        print(f"Using temporary directory: {temp_dir}")
        
        try:
            # Find all CSV files
            csv_files = list(csv_dir.glob("*.csv"))
            
            if not csv_files:
                print(f"No CSV files found in {csv_dir}")
                return 0, 0
            
            print(f"Found {len(csv_files)} CSV files")
            
            successful = 0
            failed = 0
            
            for csv_file in tqdm(csv_files, desc="Processing videos on-the-fly"):
                try:
                    # Get corresponding video file
                    video_name = csv_file.stem
                    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
                    
                    video_file = None
                    for ext in video_extensions:
                        potential_video = video_dir / f"{video_name}{ext}"
                        if potential_video.exists():
                            video_file = potential_video
                            break
                    
                    if not video_file:
                        logging.warning(f"Video not found for CSV: {csv_file.name}")
                        failed += 1
                        continue
                    
                    # Check if features already exist
                    feature_output_path = output_dir / f"{video_name}.npy"
                    if feature_output_path.exists():
                        logging.info(f"Features already exist for {video_name}, skipping...")
                        successful += 1
                        continue
                    
                    # Extract peak frame to temp directory
                    peak_result = self.peak_extractor.process_single_video(
                        video_file, csv_file, temp_dir
                    )
                    
                    if peak_result["status"] != "success":
                        logging.error(f"Failed to extract peak frame for {video_name}: {peak_result}")
                        failed += 1
                        continue
                    
                    # Load the extracted peak frame
                    peak_frame_path = temp_dir / f"{video_name}_peak_frame.png"
                    
                    if not peak_frame_path.exists():
                        logging.error(f"Peak frame not found: {peak_frame_path}")
                        failed += 1
                        continue
                    
                    try:
                        peak_frame = Image.open(peak_frame_path).convert('RGB')
                    except Exception as e:
                        logging.error(f"Failed to load peak frame {peak_frame_path}: {str(e)}")
                        failed += 1
                        continue
                    
                    # Extract features
                    features = self.extract_features(peak_frame)
                    if features is None:
                        failed += 1
                        continue
                    
                    # Save features
                    np.save(feature_output_path, features)
                    successful += 1
                    
                    # Cleanup peak frame if requested
                    if cleanup and peak_frame_path.exists():
                        peak_frame_path.unlink()
                    
                    if successful % 50 == 0:  # Log every 50 videos
                        logging.info(f"Processed {successful} videos, current: {video_name} -> {features.shape}")
                
                except Exception as e:
                    logging.error(f"Failed to process {video_name}: {str(e)}")
                    failed += 1
            
            return successful, failed
            
        finally:
            # Cleanup temp directory if we created it
            if temp_created and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logging.warning(f"Failed to cleanup temp directory: {e}")
    
    def process_peak_frames(self, **kwargs):
        """
        Main processing method that routes to appropriate processing function.
        """
        use_on_the_fly = kwargs.get('use_on_the_fly', False)
        
        if use_on_the_fly:
            # On-the-fly processing
            video_dir = kwargs.get('video_dir')
            csv_dir = kwargs.get('csv_dir')
            output_dir = kwargs.get('output_dir')
            temp_dir = kwargs.get('temp_dir')
            cleanup = kwargs.get('cleanup', True)
            
            if not video_dir or not csv_dir:
                raise ValueError("video_dir and csv_dir are required for on-the-fly processing")
            
            print("Processing with on-the-fly peak frame extraction...")
            successful, failed = self.process_peak_frames_on_the_fly(
                video_dir, csv_dir, output_dir, temp_dir, cleanup
            )
        else:
            # Directory-based processing
            peak_frames_dir = kwargs.get('peak_frames_dir')
            output_dir = kwargs.get('output_dir')
            
            if not peak_frames_dir:
                raise ValueError("peak_frames_dir is required for directory-based processing")
            
            print("Processing pre-extracted peak frames from directory...")
            successful, failed = self.process_peak_frames_from_directory(
                peak_frames_dir, output_dir
            )
        
        # Print final results
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total processed: {successful + failed}")
        print(f"Features saved to: {kwargs.get('output_dir')}")
        print(f"Feature format: [1025, 1408] - CLS token + 1024 patches, 1408-dim each")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Extract EVA-ViT features from peak frame images")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted features")
    parser.add_argument("--weights_dir", default="D:\Acads\BTP\preprocessing_code\models_weights", help="Directory to download/store model weights")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="Model precision")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--peak_frames_dir", help="Directory containing pre-extracted peak frame images")
    mode_group.add_argument("--on_the_fly", action="store_true", help="Extract peak frames on-the-fly")
    
    # On-the-fly mode arguments
    parser.add_argument("--video_dir", help="Directory containing video files (required for on-the-fly)")
    parser.add_argument("--csv_dir", help="Directory containing OpenFace CSV files (required for on-the-fly)")
    parser.add_argument("--temp_dir", help="Temporary directory for peak frames (optional, auto-created if not provided)")
    parser.add_argument("--no_cleanup", action="store_true", help="Don't delete temporary peak frames after processing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.on_the_fly:
        if not args.video_dir or not args.csv_dir:
            parser.error("--video_dir and --csv_dir are required when using --on_the_fly")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'eva_extraction_{args.log_level.lower()}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Validate directories
    if args.peak_frames_dir and not os.path.exists(args.peak_frames_dir):
        print(f"Error: Peak frames directory does not exist: {args.peak_frames_dir}")
        return
    
    if args.on_the_fly:
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory does not exist: {args.video_dir}")
            return
        if not os.path.exists(args.csv_dir):
            print(f"Error: CSV directory does not exist: {args.csv_dir}")
            return
    
    # Check if CUDA is available and adjust precision accordingly
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
        args.precision = "fp32"  # Force fp32 on CPU
    elif args.device == "cpu":
        print("Running on CPU, forcing fp32 precision")
        args.precision = "fp32"  # Always use fp32 on CPU
    
    print(f"Configuration:")
    if args.on_the_fly:
        print(f"  Mode: On-the-fly peak frame extraction")
        print(f"  Video dir: {args.video_dir}")
        print(f"  CSV dir: {args.csv_dir}")
        print(f"  Temp dir: {args.temp_dir or 'Auto-created'}")
        print(f"  Cleanup: {not args.no_cleanup}")
    else:
        print(f"  Mode: Pre-extracted peak frames")
        print(f"  Peak frames dir: {args.peak_frames_dir}")
    
    print(f"  Output dir: {args.output_dir}")
    print(f"  Weights dir: {args.weights_dir}")
    print(f"  Device: {args.device}")
    print(f"  Precision: {args.precision}")
    
    # Initialize extractor
    try:
        extractor = EVAViTFeatureExtractor(
            device=args.device, 
            precision=args.precision,
            weights_dir=args.weights_dir
        )
    except Exception as e:
        logging.error(f"Failed to initialize extractor: {str(e)}")
        return
    
    # Process peak frames
    try:
        if args.on_the_fly:
            extractor.process_peak_frames(
                use_on_the_fly=True,
                video_dir=args.video_dir,
                csv_dir=args.csv_dir,
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                cleanup=not args.no_cleanup
            )
        else:
            extractor.process_peak_frames(
                use_on_the_fly=False,
                peak_frames_dir=args.peak_frames_dir,
                output_dir=args.output_dir
            )
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return

if __name__ == "__main__":
    main()