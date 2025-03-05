import timm
import torch.nn as nn

def get_model(model_name, num_classes=9):
    """
    Load a model (e.g., Inception-ResNet-v2, ResNet18, ViT) based on the model name provided.
    
    Args:
    - model_name: The name of the model to load (e.g., 'inception_resnet_v2', 'resnet18', 'vit_base_patch16_224').
    - num_classes: Number of output classes.
    
    Returns:
    - model: A modified pretrained model suitable for classification.
    """
    # Create the model using timm, with pre-trained weights
    model = timm.create_model(model_name, pretrained=True)

    # Modify the final classification layer based on the model architecture
    if 'resnet18' in model_name:
        # For ResNet (e.g., resnet18)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'vit' in model_name:
        # For Vision Transformers (e.g., vit_base_patch16_224)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    elif 'inception_resnet_v2' in model_name:
        # For Inception-ResNet-v2
        num_ftrs = model.classif.in_features
        model.classif = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported yet.")
    
    return model

