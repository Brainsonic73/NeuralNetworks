# %% [markdown]
# ## Part 1: Setup and Data Loading
#
# Load Fashion-MNIST with torchvision, split 60k → 50k train / 10k val,
# and normalize with mean 0.2860, std 0.3530 (Fashion-MNIST global stats).

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ### Data loading and splitting
#
# We use the standard Fashion-MNIST normalization values:
# mean = 0.2860, std = 0.3530 (computed across the training set).

# %%
# Transform: convert to tensor + normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Download training set (60,000 images)
full_train = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)

# Download test set (10,000 images) — will be used as a held-out set
test_set = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# Split 60k → 50k train / 10k val
train_set, val_set = random_split(
    full_train, [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train set: {len(train_set)} samples")
print(f"Validation set: {len(val_set)} samples")
print(f"Test set: {len(test_set)} samples")

# %% [markdown]
# ### DataLoaders
#
# Batch size 64 for the baseline. We'll see how batch size affects
# performance in Part 3.

# %%
BATCH_SIZE = 64


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# Inspect one batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Pixel value range: [{images.min():.3f}, {images.max():.3f}]")

# %% [markdown]
# ### Class names

# %%
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# %% [markdown]
# ### Visualise some samples

# %%
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    img, lbl = train_set[i]
    # Undo normalization for display
    img_display = img.squeeze() * 0.3530 + 0.2860
    ax.imshow(img_display, cmap='gray')
    ax.set_title(CLASS_NAMES[lbl], fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=100)
plt.show()
print("Saved sample_images.png")
