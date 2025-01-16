import timm
import torch.nn as nn

def get_model(num_classes=9):
    """
    Load the pretrained Inception-ResNet-v2 model and modify it for classification.
    Args:
    - num_classes: Number of output classes.
    """
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    num_ftrs = model.classif.in_features
    model.classif = nn.Sequential(
        nn.Dropout(p=0.8),
        nn.Linear(num_ftrs, num_classes)
    )
    return model
