import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from src.model import get_model  # Import the updated model

# Load hyperparameters from a YAML file
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Function to load the model checkpoint if it exists
def load_checkpoint(model, optimizer, save_dir, device):
    if not os.path.exists(save_dir):
        print(f"No checkpoint directory found at {save_dir}, starting from scratch.")
        return 0

    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print(f"No checkpoint found in {save_dir}, starting from scratch.")
        return 0

    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(save_dir, latest_checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}.")
    return epoch

# Train function
def train(model, train_loader, criterion, optimizer, epoch, writer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = total_correct / total_samples
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
    progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = total_correct / total_samples
    writer.add_scalar('Loss/validation', epoch_loss, epoch)
    writer.add_scalar('Accuracy/validation', epoch_acc, epoch)
    print(f"Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# Function to get a subset of the dataset for quick testing
def get_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

def main(config_path):
    hyperparams = load_hyperparameters(config_path)

    project_dir = hyperparams['project_dir']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    num_epochs = hyperparams['num_epochs']
    image_size = hyperparams['image_size']
    num_classes = hyperparams['num_classes']
    eta_min = hyperparams.get('eta_min', 0.0001)  # Minimum learning rate for scheduler

    quick_test = hyperparams.get('quick_test', False)
    quick_test_size = hyperparams.get('quick_test_size', 100)

    checkpoint_dir = os.path.join(project_dir, 'models', 'checkpoints')
    train_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'train')
    val_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'val')
    log_dir = os.path.join(project_dir, 'runs', 'distortion_classification_experiment')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Apply quick test subset if enabled
    if quick_test:
        print(f"Quick test mode enabled with subset size: {quick_test_size}")
        train_dataset = get_subset(train_dataset, quick_test_size)
        val_dataset = get_subset(val_dataset, quick_test_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Add Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir, device)

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, writer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, writer, device)

        # Step the scheduler after each epoch
        scheduler.step()

        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train distortion classification model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    args = parser.parse_args()
    main(args.config)
