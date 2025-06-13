import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import pandas as pd
import json
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
# from custom_dataset import CustomImageDataset
from src.custom_dataset_no_label import ImageFolderWithoutLabels 


from src.model import get_model  # Import your model definition


# Function to load a specific model checkpoint
def load_checkpoint(model, model_path, device='cpu'):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load checkpoint (map_location ensures compatibility with CPU/GPU)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Check if the checkpoint contains the model state_dict
    if 'model_state_dict' in checkpoint:
        # Case where the checkpoint contains full information (e.g., saved with optimizer and epoch)
        state_dict = checkpoint['model_state_dict']
        print(f"Model state_dict loaded from checkpoint: {model_path}")
    else:
        # Case where only the model state_dict was saved
        state_dict = checkpoint
        print(f"Model loaded from a state_dict-only checkpoint: {model_path}")

    # Handle case where model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model

def load_test_data(test_dir, image_size, batch_size, class_mapping=None):
    # Function to load the test dataset and apply the class mapping if needed
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = ImageFolderWithoutLabels(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader

# Function to evaluate the model on the test dataset
def evaluate(model_paths, test_dir, image_size, num_classes, batch_size, device, model_name, output_file):
    test_dataset, test_loader = load_test_data(test_dir, image_size, batch_size)
    
    for model_path in model_paths:
        model = get_model(num_classes=num_classes, model_name=model_name)
        model = load_checkpoint(model, model_path, device=device)
        model = model.to(device)
        
    model.eval()
    filenames = []
    softmax_probs = []
    latent_vectors = []
    labels_list = []

    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)  # Add tqdm for progress bar

    with torch.no_grad():
        for inputs, paths, labels in progress_bar:  # Dataset now returns inputs and paths
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)  # Compute softmax probabilities
            
            filenames.extend(paths)  # Collect file paths
            softmax_probs.extend(probabilities.cpu().numpy())  # Collect softmax probabilities
            
            outputs = outputs.view(outputs.size(0), -1)  # Flatten if necessary
            latent_vectors.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    
    # Save results to a CSV file
    results = pd.DataFrame({
        "Filename": filenames,
        "Softmax Probabilities": [list(prob) for prob in softmax_probs]
    })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='MC Dropout for Uncertainty Quantification in Distortion Classification')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to the trained model files')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size for the model')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for test dataloader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the model, default=resnet18')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv', help='Output CSV file for evaluation results')
    parser.add_argument('--class_mapping', type=str, help='Optional: Path to JSON file containing class mapping')
    args = parser.parse_args()
    
    # Load the class mapping from JSON
    if args.class_mapping:
        with open(args.class_mapping, 'r') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None
    
    # Perform evaluation
    evaluate(
        model_paths=args.model_paths,
        test_dir=args.test_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()