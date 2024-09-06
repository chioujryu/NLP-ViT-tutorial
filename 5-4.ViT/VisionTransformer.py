# Reference : https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6

# Import necessary PyTorch modules
from httpx import patch
import torch
import torch.nn as nn  # This module contains neural network components like layers
import torchvision.transforms as T  # Used to apply transformations to images
from torch.optim import Adam  # Adam optimizer for training the model
from torchvision.datasets.mnist import MNIST  # MNIST dataset, a common dataset for digit classification
from torch.utils.data import DataLoader  # DataLoader is used to load the dataset in batches
import numpy as np  # Import NumPy for numerical operations


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size 
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels

        self.linear_project = nn.Conv2d(in_channels=self.n_channels,
                                         out_channels=self.d_model, 
                                         kernel_size=self.patch_size, 
                                         stride=self.patch_size)
        
    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row

    def forward(self, x):

        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

        x = x.transpose(-2, -1) # (B, d_model, P) -> (B, P, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            # Expand to have class token for every image in batch
            # Expand cls_token to match the batch size and concatenate it to each input sequence
            # The token shape will become (batch_size, 1, d_model)
            tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

            # Adding class tokens to the beginning of each embedding
            # Concatenate the classification token to the beginning of the input sequence
            # This adds an additional token to each sequence for classification
            # Resulting shape will be (batch_size, seq_length+1, d_model)
            x = torch.cat((tokens_batch,x), dim=1)

            # Add positional encoding to embeddings
            # Add the positional encodings to the input sequence embeddings
            # This gives each position in the sequence a unique encoding based on its index
            x = x + self.pe

            return x

class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)