import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F

import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

from src.model import get_model  # Import the updated model
from src.utils import load_hyperparameters, save_best_model, load_checkpoint, set_seed, print_mc_dropout_results
    

def enable_dropout(model):
    """
    Enable dropout layers during test-time to perform Monte Carlo sampling.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Dropout layers should be active during MC Dropout
            module.p = 0.5
            print('dropout enabled')
    return model


def mc_dropout_predict(model, dataloader, num_classes, num_samples=100, device='cuda'):
    """
    Perform MC Dropout during validation to quantify uncertainty.
    
    Args:
    - model: The trained model.
    - dataloader: DataLoader for the validation/test data.
    - num_classes: Number of output classes.
    - num_samples: Number of forward passes (MC Dropout samples).
    - device: Device to perform computation on.

    Returns:
    - all_preds: Averaged predictions across multiple forward passes.
    - all_uncertainties: Uncertainties (variance) in predictions.
    """
    model = model.to(device)
    model = enable_dropout(model)  # Enable dropout during validation
    model.eval()  # But keep the model in evaluation mode

    all_preds = []
    all_uncertainties = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="MC Dropout Validation"):
        inputs = inputs.to(device)
        all_labels.append(labels.cpu().numpy())  # Store ground truth labels

        # Run multiple forward passes
        preds = torch.zeros(num_samples, inputs.size(0), num_classes).to(device)
        for i in range(num_samples):
            with torch.no_grad():
                outputs = model(inputs)
                preds[i] = F.softmax(outputs, dim=1)

        # Mean prediction and uncertainty estimation
        mean_preds = preds.mean(dim=0)
        uncertainties = preds.var(dim=0)

        all_preds.append(mean_preds.cpu().numpy())
        all_uncertainties.append(uncertainties.cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_uncertainties, axis=0), np.concatenate(all_labels, axis=0)


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
    seed = hyperparams.get('seed', 42)  # Seed for reproducibility
    model_name = hyperparams.get('model_name', 'resnet18')  # Default to resnet18 if not provided
    experiment_name = hyperparams.get('experiment_name', f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")  # Set experiment name from config or timestamp

    quick_test = hyperparams.get('quick_test', False)
    quick_test_size = hyperparams.get('quick_test_size', 100)

    # Create experiment-specific directories
    experiment_dir = os.path.join(project_dir, 'experiments', experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, 'models', 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'runs')

    train_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'train')
    val_dir = os.path.join(project_dir, 'dataset', 'processed_frames', 'val')

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
        
        # Training phase
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, writer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, writer, device)
        
        # Perform MC Dropout after each epoch
        mean_preds, uncertainties, labels = mc_dropout_predict(model, val_loader, num_classes=num_classes, num_samples=50, device=device)
        
        accuracy, brier_score, nll, entropy, ece = print_mc_dropout_results(mean_preds, uncertainties, labels, num_classes)
        
        writer.add_scalar('MC dropout/ Accuracy ', accuracy, epoch)
        writer.add_scalar('MC dropout/ Prediction Variance (uncertainty) ', np.mean(uncertainties), epoch)
        writer.add_scalar('MC dropout/ Brier Score ', brier_score, epoch)
        writer.add_scalar('MC dropout/ Negative Log Likelihood (NLL) ', nll, epoch)
        writer.add_scalar('MC dropout/ Predictive Entropy ', entropy, epoch)
        writer.add_scalar('MC dropout/ Expected Calibration Error (ECE) ', ece, epoch)
        
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
