import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.embedding_dim)
    

class PositinalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape(max_length <sequnce length>, embedding_dim)
        pe = torch.zeros(max_length, embedding_dim)
        # Create a vector of shape (embedding_dim)
        position = torch.arange(0,max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)) # (embedding_dim / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / embedding_dim))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / embedding_dim))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, embedding_dim)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)





        