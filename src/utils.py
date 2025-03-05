import os
import torch
import yaml
import random
from torch.utils.data import Subset
import numpy as np

def calculate_accuracy(output, target):
    """
    Calculate the accuracy of the model's predictions.
    Args:
    - output: Model predictions (logits).
    - target: Ground truth labels.
    """
    _, pred = torch.max(output, dim=1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)

def log(message):
    """
    Simple logging function to print messages.
    """
    print(message)
    
# Load hyperparameters from a YAML file
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Function to save the best model
def save_best_model(model, optimizer, epoch, save_dir, best=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    
    torch.save(state, model_path)
    
    if best:
        best_model_path = os.path.join(save_dir.split('checkpoints')[0], "best_model.pth")
        torch.save(state, best_model_path)
        print(f"Best model saved at {best_model_path}")
    else:
        print(f"Model checkpoint saved at {model_path}")
    
# Function to load the model checkpoint if it exists
def load_checkpoint(model, optimizer, save_dir, device):
    if not os.path.exists(save_dir):
        print(f"No checkpoint directory found at {save_dir}, starting from scratch.")
        return 0

    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        print(f"No checkpoint found in {save_dir}, starting from scratch.")
        return 0

    # Sort checkpoint files by epoch number (extract the epoch number from the filename)
    checkpoint_files.sort(key=lambda f: int(f.split('_')[2].split('.')[0]))
    latest_checkpoint = checkpoint_files[-1]
    checkpoint_path = os.path.join(save_dir, latest_checkpoint)

    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}.")
    return epoch

# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch (GPU)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU if applicable
    
    # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # For reproducibility

    print(f"Seed set to: {seed}")
    
# Function to get a subset of the dataset for quick testing
def get_subset(dataset, subset_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

# def calculate_brier_score(preds, labels, num_classes):
#     """
#     Calculate Brier score for a set of predictions and true labels.
#     """
#     one_hot_labels = np.eye(num_classes)[labels]
#     return np.mean(np.sum((preds - one_hot_labels) ** 2, axis=1))

def calculate_brier_score(preds, labels, num_classes, present_classes=None):
    """
    Calculate Brier score for a set of predictions and true labels.
    
    Args:
    - preds: Array of shape (N, num_classes) containing predicted probabilities for each class.
    - labels: Array of shape (N,) containing the true labels (class indices).
    - num_classes: Total number of classes in the model.
    - present_classes: List or set of classes present in the test set.
    
    Returns:
    - Brier score (float)
    """
    # Create one-hot encoded labels based on num_classes
    one_hot_labels = np.eye(num_classes)[labels]
    if present_classes:
        # Zero out probabilities for classes not present in the test set
        for class_idx in range(num_classes):
            if class_idx not in present_classes:
                preds[:, class_idx] = 0
                one_hot_labels[:, class_idx] = 0

        # Normalize the predictions so they still sum to 1 after zeroing out missing classes
        preds = preds / preds.sum(axis=1, keepdims=True)
    
    # Compute the Brier score
    brier_score = np.mean(np.sum((preds - one_hot_labels) ** 2, axis=1))
    
    return brier_score





def calculate_nll(preds, labels):
    """
    Calculate Negative Log Likelihood (NLL) for a set of predictions.
    """
    return -np.mean(np.log(preds[np.arange(len(labels)), labels] + 1e-9))


def calculate_predictive_entropy(preds):
    """
    Calculate predictive entropy for a set of predictions.
    """
    return -np.mean(np.sum(preds * np.log(preds + 1e-9), axis=1))


def calculate_ece(preds, labels, num_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    """
    confidences = np.max(preds, axis=1)
    predicted_classes = np.argmax(preds, axis=1)
    accuracies = predicted_classes == labels

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_accuracy = accuracies[bin_mask].mean() if bin_mask.any() else 0
        bin_confidence = confidences[bin_mask].mean() if bin_mask.any() else 0
        bin_prob = bin_mask.mean()

        ece += bin_prob * np.abs(bin_confidence - bin_accuracy)

    return ece

def print_mc_dropout_results(mean_preds, uncertainties, labels, num_classes=9):
    """
    MC dropout summary, also calculate Brier score, NLL, Predictive Entropy, and ECE.
    """
    predicted_classes = np.argmax(mean_preds, axis=1)
    accuracy = np.mean(predicted_classes == labels)
    
    # Calculate metrics
    brier_score = calculate_brier_score(mean_preds, labels, num_classes)
    nll = calculate_nll(mean_preds, labels)
    entropy = calculate_predictive_entropy(mean_preds)
    ece = calculate_ece(mean_preds, labels)

    # print("\n### MC Dropout Predictions: Highlighting Failing Cases ###")
    # failing_cases = []
    # for i in range(len(labels)):
    #     if predicted_classes[i] != labels[i]:
    #         failing_case = {
    #             "Sample": i + 1,
    #             "Predicted": predicted_classes[i],
    #             "True": labels[i],
    #             "Confidence": np.max(mean_preds[i]),
    #             "Uncertainty": uncertainties[i].mean()  # Mean uncertainty for simplicity
    #         }
    #         failing_cases.append(failing_case)
    #         print(f"Sample {i+1}:")
    #         print(f"  Predicted: {predicted_classes[i]} (Confidence: {np.max(mean_preds[i]):.4f}), True: {labels[i]}")
    #         print(f"  Uncertainty: {uncertainties[i].mean():.4f}")

    print(f"\n### Overall Results ###")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Uncertainty (All Samples): {np.mean(uncertainties):.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Negative Log Likelihood (NLL): {nll:.4f}")
    print(f"Predictive Entropy: {entropy:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    return accuracy, brier_score, nll, entropy, ece