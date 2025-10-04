import pandas as pd
import cv2
import subprocess
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PeakFrameExtractor:
    """Standalone class for extracting peak frames from OpenFace AU data"""
    
    def __init__(self, use_opencv: bool = True):
        """
        Initialize the peak frame extractor
        
        Args:
            use_opencv: If True, use OpenCV for frame extraction. If False, use FFmpeg.
        """
        self.use_opencv = use_opencv
        
    def load_au_data(self, csv_path: Path) -> pd.DataFrame:
        """Load and process OpenFace AU data"""
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            if df.empty:
                raise ValueError(f"Empty CSV file: {csv_path}")
            
            # Find AU intensity columns (ending with '_r')
            au_intensity_cols = [c for c in df.columns if c.endswith("_r")]
            if not au_intensity_cols:
                raise ValueError(f"No AU intensity columns found in: {csv_path}")
            
            # Calculate overall intensity (sum of all AUs)
            df["overall_intensity"] = df[au_intensity_cols].sum(axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading AU data from {csv_path}: {e}")
            raise
    
    def find_peak_frame(self, df: pd.DataFrame) -> Dict:
        """Find the frame with maximum AU intensity"""
        peak_frame_idx = df["overall_intensity"].idxmax()
        peak_frame_data = df.loc[peak_frame_idx]
        
        # Get active AUs (intensity > threshold)
        active_aus = []
        threshold = 0.5
        for col in df.columns:
            if col.endswith("_r"):
                intensity = peak_frame_data[col]
                if intensity > threshold:
                    au_name = col.replace("_r", "")
                    active_aus.append({"au": au_name, "intensity": float(intensity)})
        
        peak_info = {
            "frame_number": int(peak_frame_data["frame"]),
            "timestamp": float(peak_frame_data["timestamp"]),
            "overall_intensity": float(peak_frame_data["overall_intensity"]),
            "active_aus": sorted(active_aus, key=lambda x: x["intensity"], reverse=True)
        }
        
        return peak_info
    
    def extract_frame_opencv(self, video_path: Path, timestamp: float, output_path: Path) -> bool:
        """Extract frame using OpenCV"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate target frame number
            target_frame = int(timestamp * fps)
            
            if target_frame >= total_frames:
                logger.warning(f"Target frame {target_frame} exceeds video length {total_frames}")
                target_frame = total_frames - 1
            
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Cannot read frame at timestamp {timestamp}")
                cap.release()
                return False
            
            # Save frame
            success = cv2.imwrite(str(output_path), frame)
            cap.release()
            
            if success:
                logger.info(f"Extracted frame at {timestamp:.2f}s -> {output_path}")
                return True
            else:
                logger.error(f"Failed to save frame to {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"OpenCV extraction error: {e}")
            return False
    
    def extract_frame_ffmpeg(self, video_path: Path, timestamp: float, output_path: Path) -> bool:
        """Extract frame using FFmpeg"""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-ss", str(timestamp),
                "-vframes", "1",
                "-y",  # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Extracted frame at {timestamp:.2f}s -> {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout for {video_path}")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg or use OpenCV mode.")
            return False
        except Exception as e:
            logger.error(f"FFmpeg extraction error: {e}")
            return False
    
    def extract_frame(self, video_path: Path, timestamp: float, output_path: Path) -> bool:
        """Extract frame using the configured method"""
        if self.use_opencv:
            return self.extract_frame_opencv(video_path, timestamp, output_path)
        else:
            return self.extract_frame_ffmpeg(video_path, timestamp, output_path)
    
    def process_single_video(self, video_path: Path, csv_path: Path, output_dir: Path) -> Dict:
        """Process a single video to extract its peak frame"""
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}_peak_frame.png"
        
        # Skip if already processed
        if output_path.exists():
            logger.info(f"SKIP: {video_name} (already processed)")
            return {"status": "skipped", "video": video_name}
        
        try:
            # Load AU data and find peak
            df = self.load_au_data(csv_path)
            peak_info = self.find_peak_frame(df)
            
            # Extract frame
            success = self.extract_frame(video_path, peak_info["timestamp"], output_path)
            
            if success:
                logger.info(f"SUCCESS: {video_name} - Peak at {peak_info['timestamp']:.2f}s (intensity: {peak_info['overall_intensity']:.2f})")
                return {
                    "status": "success",
                    "video": video_name,
                    "peak_info": peak_info,
                    "output_path": str(output_path)
                }
            else:
                logger.error(f"FAILED: {video_name} - Frame extraction failed")
                return {"status": "failed", "video": video_name}
                
        except Exception as e:
            logger.error(f"ERROR: {video_name} - {e}")
            return {"status": "error", "video": video_name, "error": str(e)}
    
    def process_batch(self, video_dir: Path, csv_dir: Path, output_dir: Path) -> Dict:
        """Process all videos in a directory"""
        
        video_dir = Path(video_dir)
        csv_dir = Path(csv_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all CSV files
        csv_files = list(csv_dir.glob("*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {csv_dir}")
            return {"error": "No CSV files found"}
        
        logger.info(f"Found {len(csv_files)} CSV files")
        logger.info(f"Using {'OpenCV' if self.use_opencv else 'FFmpeg'} for frame extraction")
        
        stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0, "error": 0}
        results = []
        
        for csv_file in csv_files:
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
                logger.warning(f"Video not found for CSV: {csv_file.name}")
                continue
            
            stats["total"] += 1
            
            # Process video
            result = self.process_single_video(video_file, csv_file, output_dir)
            stats[result["status"]] += 1
            results.append(result)
            
            # Progress update
            if stats["total"] % 50 == 0:
                progress = (stats["total"] / len(csv_files)) * 100
                logger.info(f"Progress: {stats['total']}/{len(csv_files)} ({progress:.1f}%) - "
                          f"Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
        
        # Final statistics
        logger.info("="*60)
        logger.info("PEAK FRAME EXTRACTION COMPLETE")
        for key, value in stats.items():
            logger.info(f"{key.capitalize()}: {value}")
        
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info("="*60)
        
        return {"stats": stats, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Extract peak frames from videos based on OpenFace AU data")
    parser.add_argument("--video_dir", help="Directory containing video files")
    parser.add_argument("--csv_dir", help="Directory containing OpenFace CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save peak frames")
    parser.add_argument("--use_ffmpeg", action="store_true", help="Use FFmpeg instead of OpenCV")
    parser.add_argument("--single_video", help="Process only this video file (optional)")
    parser.add_argument("--single_csv", help="Process only this CSV file (optional)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PeakFrameExtractor(use_opencv=not args.use_ffmpeg)
    
    if args.single_video and args.single_csv:
        # Process single video
        video_path = Path(args.single_video)
        csv_path = Path(args.single_csv)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = extractor.process_single_video(video_path, csv_path, output_dir)
        print(f"Result: {result}")
    else:
        # Validate required arguments for batch processing
        if not args.video_dir or not args.csv_dir:
            parser.error("--video_dir and --csv_dir are required for batch processing")
        
        # Process batch
        results = extractor.process_batch(
            video_dir=args.video_dir,
            csv_dir=args.csv_dir,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    # Example usage if run directly
    if len(__import__('sys').argv) == 1:
        # Default configuration for your MELD dataset
        extractor = PeakFrameExtractor(use_opencv=True)
        
        results = extractor.process_batch(
            video_dir="D:/Acads/BTP/MELD/MELD.Raw/dev_splits_complete",
            csv_dir="D:/Acads/BTP/MELD/MELD.VideoFrames/dev_openface_aus", 
            output_dir="D:/Acads/BTP/MELD/peak_frames"
        )
    else:
        main()