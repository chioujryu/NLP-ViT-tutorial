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
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    # Obtaining Queries, Keys, and Values
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1)

    # Scaling
    attention = attention / (self.head_size ** 0.5)

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention
  

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):

        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out

# # Assuming you have created an instance of MultiHeadAttention
# multihead_attention = MultiHeadAttention(d_model=512, n_heads=8)

# # print(multihead_attention.heads)

# # Print out each attention head information
# for i, head in enumerate(multihead_attention.heads):
#     print(f"Head {i}: {head}")


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))
        print("Output after Multi-Head Attention:", out.shape)  # 打印出第一次 sub-layer 的結果

        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))
        print("Final Output after MLP:", out.shape)  # 打印出最終結果

        return out



# # 假設輸入張量的形狀為 (batch_size, sequence_length, d_model)
# x = torch.randn(32, 10, 512)

# # 初始化 Transformer Encoder
# encoder = TransformerEncoder(d_model=512, n_heads=8)

# # 前向傳播
# output = encoder(x)

class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model # Dimensionality of model
        self.n_classes = n_classes # Number of classes
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1 # 新增 [CLS] Token

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_layers)])

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):

        x = self.patch_embedding(images)

        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)
        
        x = self.classifier(x[:,0])

        return x

