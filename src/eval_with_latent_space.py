import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from src.custom_dataset import CustomImageDataset

from src.model import get_model

# Load the model
def load_checkpoint(model, model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' if the model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Function to extract features from the second-to-last layer (latent space)
def extract_latent_space(model, dataloader, device='cpu'):
    latent_vectors = []
    labels_list = []

    # Wrap the dataloader with tqdm for progress tracking
    for inputs, labels in tqdm(dataloader, desc="Extracting Latent Space"):
        inputs = inputs.to(device)

        with torch.no_grad():
            # Forward pass through the network up to the second-to-last layer
            features = model(inputs)
            features = features.view(features.size(0), -1)  # Flatten if necessary
            latent_vectors.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.concatenate(latent_vectors), np.concatenate(labels_list)

# Dimensionality reduction with t-SNE or PCA
def reduce_dimensions(latent_vectors, method='tsne'):
    if method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_vectors = tsne.fit_transform(latent_vectors)
    elif method == 'pca':
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(latent_vectors)
    return reduced_vectors

# Define a fixed colormap for the classes (adjust the colors and number of colors as per your need)
def create_colormap(num_classes):
    colors = plt.cm.tab10(np.linspace(0, 1, 9))  # Generates a colormap with distinct colors
    used_colors = [colors[1],colors[4],colors[6],colors[7],colors[8]]
    return used_colors # Mapping class index to colors

# Modified plot function to use a consistent colormap
def plot_latent_space(reduced_vectors, labels, num_classes, method_name, save_path, colormap=None):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 8))

    # If colormap is provided, use it to assign specific colors to classes
    if colormap:
        colors = [colormap[label] for label in labels]
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=colors, s=15)
    else:
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', s=15)

    plt.colorbar(scatter, ticks=range(num_classes))
    plt.title(f'2D Latent Space Visualization using {method_name.upper()}')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)

    # Save the figure
    file_name = f"{method_name}_latent_space_plot.png"
    plot_save_path = os.path.join(save_path, file_name)
    plt.savefig(plot_save_path, dpi=300)
    plt.close()  # Close the plot to avoid displaying it during the function execution

    print(f"Plot saved to {plot_save_path}")

# Load the dataset
def load_test_data(test_dir, image_size, batch_size, class_mapping=None):
    # Function to load the test dataset and apply the class mapping if needed
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomImageDataset(test_dir, transform=transform, class_mapping=class_mapping)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def main(args):
    # Load the model (modify 'get_model' to your model import)
    model = get_model(num_classes=args.num_classes, model_name=args.model_name)  
    model = load_checkpoint(model, args.model_path, args.device)

    # Load the test dataset
    test_loader = load_test_data(args.test_dir, args.image_size, args.batch_size)
    
    # Create a consistent colormap for classes
    colormap = create_colormap(args.num_classes)

    # Extract latent space vectors and corresponding labels
    latent_vectors, labels = extract_latent_space(model, test_loader, args.device)

    # Perform dimensionality reduction for t-SNE
    reduced_tsne = reduce_dimensions(latent_vectors, method='tsne')
    plot_latent_space(reduced_tsne, labels, args.num_classes, method_name='tsne', save_path=args.figure_output, colormap=colormap)

    # Perform dimensionality reduction for PCA
    reduced_pca = reduce_dimensions(latent_vectors, method='pca')
    plot_latent_space(reduced_pca, labels, args.num_classes, method_name='pca', save_path=args.figure_output, colormap=colormap)

if __name__ == '__main__':
    # Argument parser for handling input arguments
    parser = argparse.ArgumentParser(description="Latent Space Visualization Script")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing the test data')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference (default: cuda)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the DataLoader (default: 32)')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--figure_output', type=str, default='notebooks/figure/', help='Folder to store the latent space plots, default=notebooks/figure/')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the model, default=resnet18')
    parser.add_argument('--class_mapping', type=str, help='Optional: Path to JSON file containing class mapping')
  
    args = parser.parse_args()

    main(args)

