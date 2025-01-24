import os
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import numpy as np

try:
    import cupy as cp  # Optional for GPU-accelerated image processing
    gpu_available = True
except ImportError:
    gpu_available = False

def extract_and_save_frames(video_file, distortion_folder, output_dir, distortion_type, split):
    """
    Extract frames from a video file and save them to the specified directory.

    Args:
    - video_file (str): The name of the video file.
    - distortion_folder (str): The folder containing the distortion type.
    - output_dir (str): The directory to save the extracted frames.
    - distortion_type (str): The type of distortion (used for folder naming).
    - split (str): The dataset split (train/val/test).
    """
    video_path = os.path.join(distortion_folder, video_file)
    save_dir = os.path.join(output_dir, split, distortion_type)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use GPU-accelerated processing if available
        if gpu_available:
            frame = cp.array(frame)  # Convert frame to CuPy array (GPU array)
            frame = cp.asnumpy(frame)  # Convert back to NumPy array

        frame_filename = f'{os.path.splitext(video_file)[0]}_frame_{frame_idx:04d}.jpg'
        cv2.imwrite(os.path.join(save_dir, frame_filename), frame)
        frame_idx += 1

    cap.release()

def process_videos_in_parallel(video_files, distortion_folder, output_dir, distortion_type, split):
    """
    Process multiple video files in parallel to extract frames.
    """
    with ThreadPoolExecutor() as executor:
        # Submit tasks for concurrent processing
        list(tqdm(executor.map(lambda video_file: extract_and_save_frames(
                video_file, distortion_folder, output_dir, distortion_type, split), 
                video_files), 
                total=len(video_files), desc=f"Processing {split} set for {distortion_type}"))

def split_dataset_and_extract_frames(data_dir, output_dir, train_size=0.8, val_size=0.1):
    """
    Split dataset into train/val/test and extract frames from videos in parallel.
    
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

        # Process video files in parallel using ThreadPoolExecutor
        process_videos_in_parallel(train_files, distortion_folder, output_dir, distortion_type, 'train')
        process_videos_in_parallel(val_files, distortion_folder, output_dir, distortion_type, 'val')
        process_videos_in_parallel(test_files, distortion_folder, output_dir, distortion_type, 'test')

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
