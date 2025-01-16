import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse


from src.model import get_model  # Import the updated model

# Load hyperparameters from a YAML file
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Train function
def train(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Wrap DataLoader in tqdm for progress tracking
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        # Move data to the selected device (GPU/CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar description
        progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = total_correct / total_samples

    # Log to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, epoch, writer, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Wrap DataLoader in tqdm for progress tracking
    progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            # Move data to the selected device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar description
            progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = total_correct / total_samples

    # Log to TensorBoard
    writer.add_scalar('Loss/validation', epoch_loss, epoch)
    writer.add_scalar('Accuracy/validation', epoch_acc, epoch)

    print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def main(config_path):
    # Load hyperparameters from YAML file
    hyperparams = load_hyperparameters(config_path)

    # Get project directory and other hyperparameters from YAML
    project_dir = hyperparams['project_dir']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    num_epochs = hyperparams['num_epochs']
    dropout_prob = hyperparams['dropout_prob']
    image_size = hyperparams['image_size']
    num_classes = hyperparams['num_classes']

    # Dataset directories (train and val directories should already be organized into folders)
    train_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'train')
    val_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'val')
    log_dir = os.path.join(project_dir, 'runs', 'distortion_classification_experiment')

    # Prepare dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model from model.py
    model = get_model(num_classes=num_classes)

    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, writer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, writer, device)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train distortion classification model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    
    args = parser.parse_args()
    
    # Call the main function with the provided config file
    main(args.config)
