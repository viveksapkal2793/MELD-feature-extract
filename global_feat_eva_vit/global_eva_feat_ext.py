#!/usr/bin/env python3
"""
Clean EVA-ViT feature extraction script with organized dependencies.
Extracts 1408-dimensional features from peak frame images.
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dependencies import create_eva_vit_g, BlipImageEvalProcessor

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
    
    def process_peak_frames(self, peak_frames_dir, output_dir, image_extensions=None):
        """
        Process all peak frame images in directory and save features.
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
        
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total processed: {successful + failed}")
        print(f"Features saved to: {output_dir}")
        print(f"Feature format: [1025, 1408] - CLS token + 1024 patches, 1408-dim each")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Extract EVA-ViT features from peak frame images")
    parser.add_argument("--peak_frames_dir", required=True, help="Directory containing peak frame images")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted features")
    parser.add_argument("--weights_dir", default="./weights", help="Directory to download/store model weights")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"], help="Model precision")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--batch_log", default=100, type=int, help="Log every N processed images")
    
    args = parser.parse_args()
    
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
    if not os.path.exists(args.peak_frames_dir):
        print(f"Error: Peak frames directory does not exist: {args.peak_frames_dir}")
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
    extractor.process_peak_frames(args.peak_frames_dir, args.output_dir)

if __name__ == "__main__":
    main()