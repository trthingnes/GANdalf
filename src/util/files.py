import logging
import os

import torch

default_directory = "model_data"


def filename_format(name):
    return f"{name}.pt"


def save_state(model, name, directory=default_directory):
    """Saves the models state dict to a file with the given name."""
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(model.state_dict(), f"{directory}/{filename_format(name)}")
    logging.info(f"Saved file '{filename_format(name)}'")


def load_state(model, name, directory=default_directory):
    """Reads a models state dict from the file with the given name and loads it in the model."""
    model.load_state_dict(torch.load(f"{directory}/{filename_format(name)}"))
    logging.info(f"Loaded file '{filename_format(name)}'")
    return model
