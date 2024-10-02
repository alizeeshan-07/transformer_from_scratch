import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size : int):
        super()._init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # in pytorch, given a number, it will provide you with the same vector everytime
        # want to mapping between numbers and vector of size 512 (d_model)

        self.embedding = nn.Embedding(vocab_size, d_model) # Embedding provide mapping of number and vector

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # mentioned in the paper to multiply with square root of d_model
    
    # adding positional embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # to reduce overfitting

        # building a matrix for seq_len x d_model
        # we want vectors of d_model size i.e. 512 for the maximum sequence length. Sentence length or for all words

        # creating a matrix of shape (seq_len, d_model)

        pe = torch.zeros(seq_len, d_model)

        # formula used to create positional encoding is a bit complex i.e. that sin and cos formula
        # we will convert that formula with log space. We take exponential of log of a number the result is the same

        # creating a vector of shape seq_len i.e. (seq_len, 1). The vector will represent the position of the word inside the sentence

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1) # numerator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model )) # denominator

        # now apply for even and odd positions

        # even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        # adding batch dimension to the tensor so we can apply to all batch of sentence because now their shape is seq_len x d_model
        # but we have batch of sentences so we add new dimension to pe

        pe = pe.unsqueeze(0) # tensor of length (1, seq_len, d_model) # here 1 is the place holder for batch size. 
        # if batch size is 4 i.e. 4 sentences then tensor will be of (4, seq_len, d_model)

        









    



    








