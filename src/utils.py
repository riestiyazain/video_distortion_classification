import torch

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
