import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from custom_dataset import CustomImageDataset

from src.model import get_model  # Import your model definition

# Load hyperparameters from a YAML file
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

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

    test_dataset = CustomImageDataset(test_dir, transform=transform, class_mapping=class_mapping)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset, test_loader

# Function to evaluate the model on the test dataset
def evaluate(model_paths, test_dir, image_size, num_classes, batch_size, num_samples, device, model_name, class_mapping):
    test_dataset, test_loader = load_test_data(test_dir, image_size, batch_size, class_mapping=class_mapping)
    
    for model_path in model_paths:
        model = get_model(num_classes=num_classes, model_name=model_name)
        model = load_checkpoint(model, model_path, device=device)
        model = model.to(device)
        
    model.eval()
    y_true = []
    y_pred = []
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)  # Add tqdm for progress bar

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar with running accuracy
            accuracy = total_correct / total_samples
            progress_bar.set_postfix(accuracy=accuracy)

    accuracy = total_correct / total_samples
    
    # Print accuracy
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    # class_names = test_dataset.classes  # Get class labels from the dataset
    class_names = os.listdir("dataset/processed_frames/test")
    print("\nClassification Report:")
    # print(class_names)
    # print(classification_report(y_true, y_pred, target_names=np.unique(y_pred), zero_division=np.nan))
    print(classification_report(y_true, y_pred, zero_division=np.nan))

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return y_true, y_pred, accuracy

def main():
    parser = argparse.ArgumentParser(description='MC Dropout for Uncertainty Quantification in Distortion Classification')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to the trained model files')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size for the model')
    parser.add_argument('--num_classes', type=int, default=9, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for test dataloader')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of forward passes (MC samples) for each input')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the model, default=resnet18')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel processing')
    # parser.add_argument('--output_file', type=str, default='failing_cases.csv', help='Output CSV file for failing cases')
    parser.add_argument('--class_mapping', type=str, help='Optional: Path to JSON file containing class mapping')
    args = parser.parse_args()
    
    # Load the class mapping from JSON
    if args.class_mapping:
        with open(args.class_mapping, 'r') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None
    # Perform MC Dropout evaluation
    
    y_true, y_pred, accuracy = evaluate(
        model_paths=args.model_paths,
        test_dir=args.test_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
        model_name=args.model_name,
        class_mapping=class_mapping
    )
    
    
    
    
# def main(config_path, model_path):
#     # Load hyperparameters
#     hyperparams = load_hyperparameters(config_path)

#     project_dir = hyperparams['project_dir']
#     batch_size = hyperparams['batch_size']
#     image_size = hyperparams['image_size']
#     num_classes = hyperparams['num_classes']
#     model_name = hyperparams.get('model_name', 'resnet18')  # Default to resnet18 if not provided

#     test_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'test')

#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#     ])

#     # Load test dataset
#     test_dataset = datasets.ImageFolder(test_dir, transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Load the model
#     model = get_model(num_classes=num_classes, model_name=model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # Load the provided model checkpoint
#     model = load_checkpoint(model, model_path, device)

#     # Evaluate the model
#     y_true, y_pred, accuracy = evaluate(model, test_loader, device)
    
#     # Print accuracy
#     print(f"Test Accuracy: {accuracy:.4f}")

#     # Generate classification report
#     class_names = test_dataset.classes  # Get class labels from the dataset
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=class_names))

#     # Print confusion matrix
#     print("Confusion Matrix:")
#     cm = confusion_matrix(y_true, y_pred)
#     print(cm)

if __name__ == "__main__":
   main()
