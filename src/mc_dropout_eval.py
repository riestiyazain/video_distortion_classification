import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from src.model import get_model  # Assuming your model definition is in src/model.py
from src.utils import calculate_brier_score, calculate_nll, calculate_ece, calculate_predictive_entropy
from custom_dataset import CustomImageDataset
import json

def enable_dropout(model):
    """
    Enable dropout layers during test-time to perform Monte Carlo sampling.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            print('dropout enabled')


def mc_dropout_predict(model, dataloader, num_classes, num_samples=50, device='cuda'):
    """
    Perform MC Dropout and quantify uncertainty.

    Args:
    - model: The trained model.
    - dataloader: DataLoader for the validation/test data.
    - num_classes: Number of output classes.
    - num_samples: Number of forward passes (Monte Carlo samples).
    - device: Device to perform computation on.

    Returns:
    - all_preds: Averaged predictions across multiple forward passes.
    - all_uncertainties: Uncertainties (variance) in predictions.
    """
    model = model.to(device)
    model.eval()
    enable_dropout(model)  # Enable dropout layers during inference

    all_preds = []
    all_uncertainties = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Performing MC Dropout"):
        inputs = inputs.to(device)
        all_labels.append(labels.numpy())  # Store ground truth labels

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


def load_test_data(test_dir, image_size, batch_size, class_mapping=None):
    # Function to load the test dataset and apply the class mapping if needed
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomImageDataset(test_dir, transform=transform, class_mapping=class_mapping)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def load_checkpoint(model, model_path, device='cpu'):
    """
    Load model from checkpoint, handling 'DataParallel' and state_dict variations.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Check if checkpoint contains the model_state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Model state_dict loaded from checkpoint: {model_path}")
    else:
        state_dict = checkpoint
        print(f"Model loaded from a state_dict-only checkpoint: {model_path}")

    # Handle case where model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return model


def evaluate_mc_dropout(model_paths, test_dir, image_size, num_classes, batch_size, num_samples, device, model_name, class_mapping):
    """
    Evaluate models using MC Dropout, comparing results with actual labels.
    """
    test_loader = load_test_data(test_dir, image_size, batch_size, class_mapping=class_mapping)


    all_model_preds = []
    all_model_uncertainties = []

    for model_path in model_paths:
        model = get_model(num_classes=num_classes, model_name=model_name)
        model = load_checkpoint(model, model_path, device=device)
        preds, uncertainties, labels = mc_dropout_predict(model, test_loader, num_classes, num_samples, device)
        all_model_preds.append(preds)
        all_model_uncertainties.append(uncertainties)

    # Average predictions and uncertainties across models
    mean_preds = np.mean(all_model_preds, axis=0)
    mean_uncertainties = np.mean(all_model_uncertainties, axis=0)

    return mean_preds, mean_uncertainties, labels



def export_failing_cases_to_csv(failing_cases, output_file):
    """
    Export failing cases to a CSV file.
    """
    os.makedirs(output_file.split("/")[0], exist_ok=True)
    df = pd.DataFrame(failing_cases)
    df.to_csv(output_file, index=False)
    print(f"Failing cases exported to {output_file}")


def print_failing_results(mean_preds, uncertainties, labels, output_file, num_classes, class_mapping):
    """
    Print only failing cases (where prediction does not match the true label) with uncertainty.
    Also calculate Brier score, NLL, Predictive Entropy, and ECE.
    """
    predicted_classes = np.argmax(mean_preds, axis=1)
    accuracy = np.mean(predicted_classes == labels)
    
    if class_mapping:
        print('class mapping, calculating based only available classes')
        present_classes = [1,4,6,7,8]
        # Calculate metrics
        brier_score = calculate_brier_score(mean_preds, labels, num_classes, present_classes)
        nll = calculate_nll(mean_preds, labels)
        entropy = calculate_predictive_entropy(mean_preds)
        ece = calculate_ece(mean_preds, labels)
    else:
        # Calculate metrics
        brier_score = calculate_brier_score(mean_preds, labels, num_classes)
        nll = calculate_nll(mean_preds, labels)
        entropy = calculate_predictive_entropy(mean_preds)
        ece = calculate_ece(mean_preds, labels)

    print("\n### MC Dropout Predictions: Highlighting Failing Cases ###")
    failing_cases = []
    for i in range(len(labels)):
        if predicted_classes[i] != labels[i]:
            failing_case = {
                "Sample": i + 1,
                "Predicted": predicted_classes[i],
                "True": labels[i],
                "Confidence": np.max(mean_preds[i]),
                "Uncertainty": uncertainties[i].mean()  # Mean uncertainty for simplicity
            }
            failing_cases.append(failing_case)
            # print(f"Sample {i+1}:")
            # print(f"  Predicted: {predicted_classes[i]} (Confidence: {np.max(mean_preds[i]):.4f}), True: {labels[i]}")
            # print(f"  Uncertainty: {uncertainties[i].mean():.4f}")

    print(f"\n### Overall Results ###")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Uncertainty (All Samples): {np.mean(uncertainties):.8f}")
    print(f"Brier Score: {brier_score:.8f}")
    print(f"Negative Log Likelihood (NLL): {nll:.8f}")
    print(f"Predictive Entropy: {entropy:.8f}")
    print(f"Expected Calibration Error (ECE): {ece:.8f}")

    # Export failing cases to a CSV file
    export_failing_cases_to_csv(failing_cases, output_file)


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
    parser.add_argument('--output_file', type=str, default='failing_cases.csv', help='Output CSV file for failing cases')
    parser.add_argument('--class_mapping', type=str, help='Optional: Path to JSON file containing class mapping')
    args = parser.parse_args()
    
    # Load the class mapping from JSON
    if args.class_mapping:
        with open(args.class_mapping, 'r') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None
    # Perform MC Dropout evaluation
    mean_preds, uncertainties, labels = evaluate_mc_dropout(
        model_paths=args.model_paths,
        test_dir=args.test_dir,
        image_size=args.image_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
        model_name=args.model_name,
        class_mapping=class_mapping
        # num_workers=args.num_workers
    )
    
   
    # Print the failing results and return uncertainty for failing cases
    print_failing_results(mean_preds, uncertainties, labels, args.output_file, args.num_classes, class_mapping)


if __name__ == "__main__":
    main()
