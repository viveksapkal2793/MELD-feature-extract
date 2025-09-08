import os
import argparse
import pandas as pd
from tqdm import tqdm
import subprocess

def create_emotion_mapping(df, emotion_column='Emotion'):
    """Creates a mapping from emotion strings to integer IDs."""
    emotions = sorted(df[emotion_column].unique())
    emotion_to_id = {emotion: i for i, emotion in enumerate(emotions)}
    print("Created emotion mapping:")
    for emotion, eid in emotion_to_id.items():
        print(f"- '{emotion}': {eid}")
    return emotion_to_id

def get_frame_count(directory):
    """Counts the number of .bmp files in a directory."""
    return len([name for name in os.listdir(directory) if name.endswith('.bmp')])

def process_videos_with_ffmpeg(video_dir, metadata_csv, output_dir, use_gpu=True):
    """
    Extracts frames from MELD video clips using FFmpeg and creates a list file.
    """
    # 1. Setup paths and load metadata
    faces_output_dir = os.path.join(output_dir, 'all_faces')
    list_file_path = os.path.join(output_dir, 'meld_list.txt')

    os.makedirs(faces_output_dir, exist_ok=True)

    try:
        df = pd.read_csv(metadata_csv)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_csv}")
        return

    emotion_to_id = create_emotion_mapping(df)
    
    list_file_entries = []
    print(f"\nProcessing {len(df)} utterances from {metadata_csv} using FFmpeg...")

    # 2. Iterate through each utterance in the metadata
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Frames"):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        emotion = row['Emotion']

        # Construct video name and paths
        video_name = f"dia{dialogue_id}_utt{utterance_id}"
        video_file_name = f"{video_name}.mp4"
        video_file_path = os.path.join(video_dir, video_file_name)

        utterance_frame_dir = os.path.join(faces_output_dir, video_name)
        os.makedirs(utterance_frame_dir, exist_ok=True)

        if not os.path.exists(video_file_path):
            continue

        # 3. Use FFmpeg to extract frames
        frame_save_pattern = os.path.join(utterance_frame_dir, "%05d.bmp")
        
        # Base command
        command = ['ffmpeg']
        
        # Add hardware acceleration flags if use_gpu is True
        if use_gpu:
            # For NVIDIA GPUs. For AMD/Intel, you might use 'dxva2' or 'd3d11va'
            command.extend(['-hwaccel', 'cuda']) 
        
        # Input/Output and formatting command
        command.extend([
            '-i', video_file_path,
            '-y',  # Overwrite output files without asking
            frame_save_pattern
        ])
        
        # Run the command, hiding FFmpeg's verbose output
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 4. Count extracted frames and add entry for the list file
        frame_count = get_frame_count(utterance_frame_dir)
        if frame_count > 0:
            emotion_id = emotion_to_id.get(emotion, -1)
            list_file_entries.append(f"{video_name} {frame_count} {emotion_id}")

    # 5. Write the final list file
    if list_file_entries:
        print(f"\nWriting list file to {list_file_path}...")
        with open(list_file_path, 'w') as f:
            f.write('\n'.join(list_file_entries))
        print("List file created successfully.")

    print("\nPreprocessing complete.")
    print(f"Extracted frames are located in: {faces_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess MELD dataset for feature extraction using FFmpeg.")
    parser.add_argument('--video_dir', type=str, required=True, help="Path to the directory containing MELD video clips.")
    parser.add_argument('--metadata_csv', type=str, required=True, help="Path to the MELD metadata CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory where processed files will be saved.")
    parser.add_argument('--no_gpu', action='store_true', help="Disable GPU acceleration and use CPU-only FFmpeg.")
    
    args = parser.parse_args()

    process_videos_with_ffmpeg(args.video_dir, args.metadata_csv, args.output_dir, use_gpu=not args.no_gpu)