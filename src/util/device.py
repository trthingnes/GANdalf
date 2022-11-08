import torch


def get_device(allow_cuda=True, seed=None):
    """Gets CPU or CUDA device and prints versions. Will use CUDA if available and allowed."""
    use_cuda = allow_cuda and torch.cuda.is_available()

    print("PyTorch version: {}".format(torch.__version__))
    if use_cuda:
        print("Using CUDA version: {}\n".format(torch.version.cuda))
        if seed:
            torch.cuda.manual_seed(seed)
    else:
        print("Using CPU")

    return torch.device("cuda:0" if use_cuda else "cpu")
