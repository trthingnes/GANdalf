import torch.jit
import os

default_directory = "model_data"


def filename_format(name):
    return f"{name}_ts.pt"


def save_model(model, name, directory=default_directory):
    """Converts the given model into TorchScript and saves it to a .pt file."""
    # TODO: Create folder if it does not exist.
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.jit.script(model).save(f"{directory}/{filename_format(name)}")


def load_model(name, directory=default_directory):
    """Reads TorchScript from the given .pt file and returns the model saved in it."""
    return torch.jit.load(f"{directory}/{filename_format(name)}")
