import torchvision.datasets
from torch.utils.data import Dataset
from PIL import Image


class FashionMNIST(Dataset):
    def __init__(self, data_path="training_data", transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.FashionMNIST(root=data_path, transform=transform)
        self.category_labels = self.dataset.classes
        self.image_categories = self.dataset.targets
        self.images = self.dataset.data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image = Image.fromarray(self.images[i].detach().numpy())
        label = self.category_labels[int(self.image_categories[i])]

        return image, label
