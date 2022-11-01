import torch

def get_device(allow_cuda=True):
    use_cuda = allow_cuda and torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")