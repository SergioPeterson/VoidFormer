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




        