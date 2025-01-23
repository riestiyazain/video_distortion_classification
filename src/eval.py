import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import yaml
from tqdm import tqdm  # Import tqdm for progress bar

from src.model import get_model  # Import your model definition

# Load hyperparameters from a YAML file
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Function to load a specific model checkpoint
def load_checkpoint(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    return model

# Function to evaluate the model on the test dataset
def evaluate(model, test_loader, device):
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
    return y_true, y_pred, accuracy

def main(config_path, model_path):
    # Load hyperparameters
    hyperparams = load_hyperparameters(config_path)

    project_dir = hyperparams['project_dir']
    batch_size = hyperparams['batch_size']
    image_size = hyperparams['image_size']
    num_classes = hyperparams['num_classes']

    test_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'test')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = get_model(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the provided model checkpoint
    model = load_checkpoint(model, model_path, device)

    # Evaluate the model
    y_true, y_pred, accuracy = evaluate(model, test_loader, device)
    
    # Print accuracy
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    class_names = test_dataset.classes  # Get class labels from the dataset
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate distortion classification model on test data.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint to evaluate.')
    args = parser.parse_args()
    main(args.config, args.model_path)
