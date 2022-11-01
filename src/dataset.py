import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_preprocessed_dataset(data_path, x_dim):
    return datasets.MNIST(
        root=data_path,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(x_dim),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
