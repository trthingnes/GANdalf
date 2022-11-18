import torch


def get_device(allow_cuda=True, seed=None):
    """Gets CPU or CUDA device and prints versions. Will use CUDA if available and allowed."""
    use_cuda = allow_cuda and torch.cuda.is_available()

    print(f"PyTorch version: {torch.__version__}")
    if use_cuda:
        print("Using CUDA")
        if seed:
            torch.cuda.manual_seed(seed)
    else:
        print("Using CPU")
        if seed:
            torch.manual_seed(seed)

    return torch.device("cuda:0" if use_cuda else "cpu")
