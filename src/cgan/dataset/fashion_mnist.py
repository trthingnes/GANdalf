import torchvision.transforms as transforms
import torchvision.datasets as datasets


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self):
        super().__init__(
            root="training_data",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            ),
            download=True,
        )
        self.labels = self.classes
        self.n_labels = len(self.classes)
        self.img_size = self.data.data.shape[1]
