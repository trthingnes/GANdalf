import torchvision.datasets as datasets
import torchvision.transforms as transforms


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self):
        super().__init__(
            root="training_data",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )
        self.labels = self.classes
        self.n_labels = len(self.classes)
        self.img_size = self.data.data.shape[1]
