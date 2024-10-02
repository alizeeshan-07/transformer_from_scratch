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

        # register the tensor in the buffer of the moduel

        self.register_buffer('pe', pe) # tensor that you want to keep inside the model not as a learned parameter but you want it to be
        # saved when you save the file of the model. you should register it as a buffer. This way the tensor will be saved in the file
        # along with the state of the mode.

    # we just need to add this positional encoding to embedding of word inside the sentence.

    # Summary:
    # The  below code is doing the following:
    # Adds positional encodings (which are not learned, i.e., fixed) to the input embeddings x.
    # Passes the result through a dropout layer for regularization.
    # Returns the modified embeddings after dropout.
    # Example:
    # Assume x is a tensor of shape (batch_size=1, seq_len=5, d_model=6) (a sequence of length 5 with 6-dimensional token embeddings).
    # Positional encodings of shape (1, seq_len=5, d_model=6) are added to x.
    # Afterward, dropout is applied to the updated x, and the resulting tensor is returned for further layers in the model.

    def forward(self, x): # x here is the input embedding
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # positional embeddings are not learned
        return self.dropout(x)










    



    








