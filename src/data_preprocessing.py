import os
import cv2
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def split_dataset_and_extract_frames(data_dir, output_dir, train_size=0.8, val_size=0.1):
    """
    Split dataset into train/val/test and extract frames from videos.
    
    Args:
    - data_dir (str): Path to the raw dataset directory.
    - output_dir (str): Path to save processed frames.
    - train_size (float): Proportion of the data to use for training.
    - val_size (float): Proportion of the data to use for validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    distortion_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

    # Create subdirectories for train, val, test sets
    for split in ['train', 'val', 'test']:
        for distortion_type in distortion_types:
            os.makedirs(os.path.join(output_dir, split, distortion_type), exist_ok=True)

    for distortion_type in distortion_types:
        distortion_folder = os.path.join(data_dir, distortion_type)

        # Skip files that are not videos and ignore system files like .DS_Store
        video_files = [f for f in os.listdir(distortion_folder) if f.endswith(('.mp4', '.avi', '.mkv')) and not f.startswith('.')]

        # Train-test split
        train_val_files, test_files = train_test_split(video_files, test_size=1-train_size, random_state=42)
        train_files, val_files = train_test_split(train_val_files, test_size=val_size/(train_size + val_size), random_state=42)

        def extract_and_save_frames(video_file, split):
            video_path = os.path.join(distortion_folder, video_file)
            save_dir = os.path.join(output_dir, split, distortion_type)

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f'{os.path.splitext(video_file)[0]}_frame_{frame_idx:04d}.jpg'
                cv2.imwrite(os.path.join(save_dir, frame_filename), frame)
                frame_idx += 1

            cap.release()

        # Using tqdm to track progress during frame extraction
        for video_file in tqdm(train_files, desc=f"Processing train set for {distortion_type}"):
            extract_and_save_frames(video_file, 'train')
        for video_file in tqdm(val_files, desc=f"Processing validation set for {distortion_type}"):
            extract_and_save_frames(video_file, 'val')
        for video_file in tqdm(test_files, desc=f"Processing test set for {distortion_type}"):
            extract_and_save_frames(video_file, 'test')

    print("Dataset splitting and frame extraction complete!")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Split dataset and extract frames from videos.")
    
    # Arguments for input/output directories and dataset splitting
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the raw dataset directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save processed frames.")
    parser.add_argument('--train_size', type=float, default=0.8, help="Proportion of the dataset to use for training (default: 0.8).")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of the dataset to use for validation (default: 0.1).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    split_dataset_and_extract_frames(args.data_dir, args.output_dir, args.train_size, args.val_size)
