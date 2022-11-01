import torch
import torch.backends.cudnn as cudnn


def get_device(allow_cuda, seed):
    """Enables CUDA is allowed and available."""
    allow_cuda = allow_cuda and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if allow_cuda:
        print("CUDA version: {}\n".format(torch.version.cuda))

    if allow_cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    return torch.device("cuda:0" if allow_cuda else "cpu")


def weights_init(m):
    """Initializes weights based on classname of m."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
