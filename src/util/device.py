import logging

import torch


def get_device(allow_cuda=True, seed=None):
    """Gets CPU or CUDA device and prints versions. Will use CUDA if available and allowed."""
    use_cuda = allow_cuda and torch.cuda.is_available()

    logging.info(f"PyTorch version: {torch.__version__}")
    if use_cuda:
        logging.info("Using CUDA")
        if seed:
            torch.cuda.manual_seed(seed)
    else:
        logging.info("Using CPU")
        if seed:
            torch.manual_seed(seed)

    return torch.device("cuda" if use_cuda else "cpu")


def get_device_count(cuda):
    """Gets the number of CPU or CUDA devices available to be used."""
    count = torch.cuda.device_count() if cuda else torch.get_num_threads()
    logging.info(f"Using {count} {'CUDA' if cuda else 'CPU'} devices")

    return count
