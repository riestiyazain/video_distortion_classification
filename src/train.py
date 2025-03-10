import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

from src.model import get_model  # Import the updated model
from src.utils import load_hyperparameters, save_best_model, load_checkpoint, set_seed, get_subset

# Train function with tqdm for progress tracking
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

# Validation function with tqdm for progress tracking
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
    seed = hyperparams.get('seed', 42)  # Seed for reproducibility
    model_name = hyperparams.get('model_name', 'resnet18')  # Default to resnet18 if not provided
    experiment_name = hyperparams.get('experiment_name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")  # Set experiment name from config or timestamp

    quick_test = hyperparams.get('quick_test', False)
    quick_test_size = hyperparams.get('quick_test_size', 100)

    # Create experiment-specific directories
    experiment_dir = os.path.join(project_dir, 'experiments', experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'models', 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'runs')

    train_dir = hyperparams.get('train_dir', os.path.join(project_dir, 'dataset', 'processed_frames', 'train')) 
    val_dir = hyperparams.get('val_dir', os.path.join(project_dir, 'dataset', 'processed_frames', 'val'))

    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

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

    model = get_model(model_name, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Add Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir, device)

    best_val_acc = 0.0  # Initialize the best validation accuracy

    for epoch in range(start_epoch, num_epochs):
        # Print current learning rate in tqdm
        tqdm.write(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, writer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, writer, device)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save the model for the current epoch
        save_best_model(model, optimizer, epoch, checkpoint_dir)

        # Save the model if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(model, optimizer, epoch, checkpoint_dir, best=True)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model with checkpointing")
    parser.add_argument('--config', required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config)
