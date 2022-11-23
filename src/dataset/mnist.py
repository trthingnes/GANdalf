import torchvision.datasets as datasets
import torchvision.transforms as transforms


class MNIST(datasets.MNIST):
    def __init__(self):
        super().__init__(
            root="training_data",
            transform=transforms.ToTensor(),
            download=True,
        )
        self.labels = self.classes
        self.n_labels = len(self.classes)
        self.img_size = self.data.data.shape[1]
