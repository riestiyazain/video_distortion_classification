# custom_dataset.py
from torchvision import datasets, transforms

class CustomImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, class_mapping=None):
        """
        Custom dataset for ImageFolder that handles class mapping to allow concept shift.

        Args:
        - root: Path to the dataset root folder.
        - transform: Image transformations (e.g., resize, normalize, etc.).
        - class_mapping: A dictionary mapping test dataset classes to model-trained classes.
        """
        super().__init__(root, transform=transform)

        # Convert the class mapping keys to integers
        if class_mapping is not None:
            self.class_mapping = {int(k): v for k, v in class_mapping.items()}
        else:
            self.class_mapping = None

        # Filter samples to only include classes present in class_mapping
        if self.class_mapping is not None:
            self.samples = [(path, label) for path, label in self.samples if int(label) in self.class_mapping]
            self.targets = [self.class_mapping[int(label)] for path, label in self.samples]
        else:
            self.targets = [label for path, label in self.samples]

    def __getitem__(self, index):
        """
        Override the __getitem__ method to return the sample and the remapped label.

        Args:
        - index: Index of the data point.

        Returns:
        - sample: The image tensor.
        - label: The remapped label.
        """
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        # Remap the label if a mapping is provided
        if self.class_mapping is not None:
            label = self.class_mapping[int(label)]

        return sample, label
