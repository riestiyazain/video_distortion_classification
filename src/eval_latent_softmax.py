import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

from custom_dataset import CustomImageDataset
from src.model import get_model
from src.utils import extract_latent_space, reduce_dimensions, plot_latent_space, create_colormap, load_model

def load_test_data(test_dir, image_size, batch_size, class_mapping=None):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = CustomImageDataset(test_dir, transform=transform, class_mapping=class_mapping)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader

def write_softmax_csv_latent(model, dataloader, device, output_file):
    latent_vectors = []
    labels_list = []
    model.eval()
    filenames, softmax_probs = [], []
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Softmax Eval"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            filenames.extend(paths)
            softmax_probs.extend(probs.cpu().numpy())
            
            outputs = outputs.view(outputs.size(0), -1)  # Flatten if necessary
            latent_vectors.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    # Fix: concatenate lists of arrays into a single array
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    df = pd.DataFrame({
        "Filename": filenames,
        "Softmax Probabilities": [list(prob) for prob in softmax_probs]
    })
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Softmax probabilities saved to {output_file}")
    
    # Latent space extraction and visualization
    colormap = create_colormap(args.num_classes)
    
    for method in ['tsne', 'pca']:
        reduced = reduce_dimensions(latent_vectors, method=method)
        fig_path = args.figure_output
        plot_latent_space(reduced, labels_list, args.num_classes, method_name=method, save_path=fig_path, colormap=colormap)
        print(f"{method.upper()} plot saved to {fig_path}")

def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model = get_model(model_name=args.model_name, num_classes=args.num_classes)
    model = load_model(model, args.model_path, device=device)
    model = model.to(device)

    _, test_loader = load_test_data(args.test_dir, args.image_size, args.batch_size, args.class_mapping)
    write_softmax_csv_latent(model, test_loader, device, args.output_file)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--figure_output', type=str, default='notebooks/figure/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--class_mapping', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv')
    args = parser.parse_args()
    main(args)