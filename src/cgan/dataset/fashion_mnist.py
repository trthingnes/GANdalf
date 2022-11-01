import torchvision.datasets
from torch.utils.data import Dataset
from PIL import Image


class FashionMNIST(Dataset):
    def __init__(self, data_path="training_data", transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.FashionMNIST(root=data_path, transform=transform)
        self.labels = self.dataset.classes
        self.image_labels = self.dataset.targets
        self.images = self.dataset.data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        image = Image.fromarray(self.images[id].detach().numpy())
        label = self.labels[int(self.image_labels[id])]

        return image, label
