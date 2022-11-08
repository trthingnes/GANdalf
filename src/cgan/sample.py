import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import load_model

generator = load_model("generator_Tue-Nov--8-21:35:43-2022")

samples = 10
noise = torch.randn(samples, 100)
labels_g = torch.LongTensor(np.random.randint(0, 10, samples))
images_g = generator(noise, labels_g)

plt.title(f"Generated image of a {labels_g[0]}")
plt.imshow(images_g[0])
