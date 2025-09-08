import os
import argparse
import pandas as pd
import cv2
from tqdm import tqdm

def create_emotion_mapping(df, emotion_column='Emotion'):
    """Creates a mapping from emotion strings to integer IDs."""
    emotions = sorted(df[emotion_column].unique())
    emotion_to_id = {emotion: i for i, emotion in enumerate(emotions)}
    print("Created emotion mapping:")
    for emotion, eid in emotion_to_id.items():
        print(f"- '{emotion}': {eid}")
    return emotion_to_id

def process_videos(video_dir, metadata_csv, output_dir):
    """
    Extracts frames from MELD video clips and creates a list file for feature extraction.
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
    print(f"\nProcessing {len(df)} utterances from {metadata_csv}...")

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
            # tqdm.write(f"Warning: Video file not found, skipping: {video_file_path}")
            continue

        # 3. Extract frames from the video file
        cap = cv2.VideoCapture(video_file_path)
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Save frame with a consistent naming convention (e.g., 00001.jpg)
            # Note: The user's codebase seems to use .bmp, so we save as .bmp
            frame_filename = f"{frame_count + 1:05d}.bmp"
            frame_save_path = os.path.join(utterance_frame_dir, frame_filename)
            cv2.imwrite(frame_save_path, frame)
            frame_count += 1
        
        cap.release()

        # 4. Add entry for the list file
        if frame_count > 0:
            emotion_id = emotion_to_id.get(emotion, -1) # Use -1 for unknown emotions
            # The list file format is: video_name total_frames emotion_id
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
    parser = argparse.ArgumentParser(description="Preprocess MELD dataset for feature extraction.")
    parser.add_argument('--video_dir', type=str, required=True,
                        help="Path to the directory containing MELD video clips (e.g., '.../MELD.Raw/train/train_splits').")
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help="Path to the MELD metadata CSV file (e.g., '.../MELD.Raw/train_sent_emo.csv').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to the directory where processed files will be saved.")
    
    args = parser.parse_args()

    process_videos(args.video_dir, args.metadata_csv, args.output_dir)