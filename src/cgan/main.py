import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import FashionMNIST

batch_size = 32

dataset = FashionMNIST(transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Question: Why do we do this?
]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
