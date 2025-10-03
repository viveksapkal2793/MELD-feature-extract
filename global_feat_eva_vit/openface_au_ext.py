import os
import subprocess
import time
from pathlib import Path
import logging
import signal
import sys

class OpenFaceProcessor:
    def __init__(self, openface_exe, input_dir, output_dir):
        self.openface_exe = Path(openface_exe)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.stop_processing = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / 'processing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.signal_handler)
        
        if not self.openface_exe.exists():
            raise FileNotFoundError(f"OpenFace executable not found: {self.openface_exe}")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.logger.info("Received stop signal. Stopping after current video...")
        self.stop_processing = True
    
    def get_video_files(self):
        """Get all video files from input directory"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        video_files = []
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def process_single_video(self, video_file, current_num, total_num):
        """Process a single video with OpenFace"""
        video_name = video_file.name
        # FIXED: Use correct OpenFace output filename format
        csv_output = self.output_dir / f"{video_file.stem}.csv"
        
        # Skip if already processed
        if csv_output.exists():
            self.logger.info(f"[{current_num}/{total_num}] SKIP: {video_name} (already processed)")
            return "skipped"
        
        self.logger.info(f"[{current_num}/{total_num}] PROCESSING: {video_name}")
        
        cmd = [
            str(self.openface_exe),
            "-f", str(video_file),
            "-out_dir", str(self.output_dir),
            "-aus"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0:
                # Wait a moment for file to be written
                time.sleep(2)
                
                if csv_output.exists():
                    duration = time.time() - start_time
                    self.logger.info(f"[{current_num}/{total_num}] SUCCESS: {video_name} ({duration:.1f}s)")
                    return "success"
                else:
                    self.logger.error(f"[{current_num}/{total_num}] FAILED: {video_name} - CSV not created")
                    return "failed"
            else:
                error_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
                self.logger.error(f"[{current_num}/{total_num}] FAILED: {video_name} - {error_msg}")
                return "failed"
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"[{current_num}/{total_num}] TIMEOUT: {video_name}")
            return "timeout"
        except Exception as e:
            self.logger.error(f"[{current_num}/{total_num}] ERROR: {video_name} - {str(e)}")
            return "error"
    
    def process_all_videos_sequential(self):
        """Process all videos one by one"""
        video_files = self.get_video_files()
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.input_dir}")
            return
        
        self.logger.info(f"Found {len(video_files)} video files")
        self.logger.info("Starting sequential processing...")
        
        stats = {"total": len(video_files), "success": 0, "failed": 0, "skipped": 0, "timeout": 0, "error": 0}
        
        for i, video_file in enumerate(video_files, 1):
            if self.stop_processing:
                self.logger.info("Processing stopped by user")
                break
            
            result = self.process_single_video(video_file, i, len(video_files))
            stats[result] += 1
            
            # Progress summary every 50 videos
            if i % 50 == 0 or i == len(video_files):
                progress = (i / len(video_files)) * 100
                self.logger.info(f"Progress: {i}/{len(video_files)} ({progress:.1f}%) - Success: {stats['success']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
        
        # Final statistics
        self.logger.info("="*60)
        self.logger.info("PROCESSING COMPLETE")
        for key, value in stats.items():
            self.logger.info(f"{key.capitalize()}: {value}")
        
        processed = stats['total'] - stats['skipped']
        if processed > 0:
            success_rate = (stats['success'] / processed) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info("="*60)
        
        return stats


def main():
    # ============= CONFIGURATION =============
    OPENFACE_EXE = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    INPUT_DIR = r"D:\Acads\BTP\MELD\MELD.Raw\dev_splits_complete"
    OUTPUT_DIR = r"D:\Acads\BTP\MELD\MELD.VideoFrames\dev_openface_aus"
    # ========================================
    
    try:
        processor = OpenFaceProcessor(
            openface_exe=OPENFACE_EXE,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR
        )
        
        print(f"Starting OpenFace AU extraction...")
        print(f"Input: {INPUT_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Logs: {OUTPUT_DIR}\\processing.log")
        print("Press Ctrl+C to stop gracefully...")
        
        stats = processor.process_all_videos_sequential()
        print("\nProcessing completed!")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())