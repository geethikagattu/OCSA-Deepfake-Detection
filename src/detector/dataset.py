from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DeepfakeImageFolder(datasets.ImageFolder):
    """ImageFolder with explicit labels: real=0 and fake=1."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        class_name = self.classes[label]

        if class_name == "real":
            label = 0
        elif class_name == "fake":
            label = 1
        else:
            raise ValueError(f"Unexpected class: {class_name}")

        return image, label


def create_dataloader(
    data_dir,
    batch_size=32,
    image_size=224,
    shuffle=True,
    num_workers=2,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = DeepfakeImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader
