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


# now we will proceed for layer normalization
# Layer normalization is needed to stabilize and 
# accelerate training by normalizing the inputs across the features for 
# each data point (i.e., each sequence in a transformer). It helps ensure that 
# the input values have a consistent scale, which prevents the model from becoming 
# unstable and helps the model converge faster. By normalizing the inputs to have zero mean and 
# unit variance, layer normalization reduces internal covariate shift, allowing the model to better 
# handle changes in the distribution of inputs across layers.


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None: # to avoid the zero in the denominator
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1)) # nn.Parameter means learnable parameter # multiply #scalar
        self.bias = nn.Parameter(torch.zeros(1)) # added # scalar

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) 
        # dim = -1 to keep the last dimension after the batch #mean cancels the dimension but we 
        # want to keep it that is why we used keepdim = True
        # This line calculates the mean of the tensor x along the last dimension (dim = -1), and 
        # the result is kept in the same dimensionality using keepdim = True. 
        # This ensures that the shape of the tensor remains compatible for further operations, like broadcasting.
        # x = tensor([[1.0, 2.0, 3.0],
        #     [4.0, 5.0, 6.0]])
        # after mean
        # tensor([[2.0],
        #[5.0]])
        # (2,1) instead of (2,)


        std = x.std(dim = -1, keepdim = True)
        return self.aplha * (x - mean) / (std + self.eps) + self.bias
    
    # Feed Forward Layer  -- Fully Connected Layer that model will use in both encoder and decoder
    # in paper, there are two matrices W1 and W2 multiplied with one another with ReLU in betwee
    # FFN(x) = max(0, xW1+b1)W2 + b2 
    # W1 --> d_model x dff and W2 --> dff x d_model

class FeedForwardBlock(nn.Module): # we use linear method for fully connected layers

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1 #why B1 because B1 is already defined in linear method
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self, x):
        # we have (batch, seq_len, d_model) we need --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        # we increase the dimensionality and then decrease it so the model can learn complex patterns
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# now we will implement the multi-head attention. 
class MultiHeadAttentionBlock(nn.Module):
    def _init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0,  # making sure that d_model is divisible by h d_model/h = d_k

        self.d_k = d_model/h

        # Defining weight matrices
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_v = nn.Linear(d_model, d_model) # Wo because h*dv = d_model

        self.dropout = nn.Dropout(dropout)

    @staticmethod # you can call below attention function without having an instance of MultiHeadAttention class. You can call directly as MultiHeadAttentionBlock.attention with calling the instance
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, g, seq_len, d_k) --> (batch, h, seq_len, seq_len)

        attention_scores = (query @ key.tranpose(-2,-1)) / math.sqrt(d_k) # transpose to make d_k x seq_len # @ for matrix multiplication in pytorch
        # now we will apply the mask
        if mask is not None: # if mask is defined
            attention_scores.masked_fill_(mask == 0, -1e9) #when mask==0 is true then replace by -1e9 >> -10^9
        # we want mask when we don't want to words to see the future like in decoder
        # and also dont want the padding values to interact with other values because they are just filler values
        attention_scores = attention_scores.softmax(dim = -1) # (batch,h, seq_len, seq_len) # sum of all prob = 1 for each row

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask): # mask > if we want a word or words to not interact with other words, we mask it
    # we replace their values with something very small before we apply softwax, and softmax makes it to very close to zero
    # we hide the attention between two words
    # now we apply A linear transformation is applied to the query tensor using the learned weight matrix w_q.
        # self.w_q(q) projects the input query tensor into a new space where it can be compared with the key tensor. 
        # This is part of the attention mechanism's process of determining the similarity between queries and keys.
        # In the multi-head attention mechanism, the query (q), key (k), and value (v) matrices need to be projected into new spaces using learned weight matrices (w_q, w_k, and w_v). These weight matrices are part of the model’s parameters and are learned during training.
        # The operation self.w_q(q) refers to applying a linear transformation to the input q using the learned weight matrix w_q.
        # This is essentially a fully connected layer applied to the input query tensor q.
        # w_q(q) applies a linear transformation to the input query tensor q, projecting 
        # it into a new space where it can be compared to the key and value tensors during the attention mechanism.
        
        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (batch, seq_len, d_model) i.e. Q'
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (batch, seq_len, d_model) i.e. K'
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (batch, seq_len, d_model) i.e. V'

        # now we want to split each matrix to equal head. we only want to split embeddings into "h" parts. We are not
        # splitting sentence

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # we transpose bcz we prefer
        # to have the second dimension instead of third dimension so each head must be seq_len x d_k
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # now we have all these smaller matrices, we need to calcuate the attention

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # here x is output

        #(batch, h, seq_len, d_k) -- > (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # reversing to the original dimension
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.h * self.d_k) # h*d_k is d_model  # pytorch to transform a tensor needs memory to be contigous so it can do it in place
        
        # (batch, seq_len, d_model ) --> (batch, seq_len, d_model)
        # multiplying x with Wo
        return self.w_o(x) 


    # now lets build residual connections or skip connections
    # skip connections are between add & Norm and previous layer
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer): # sublayer is previous layer
        return x + self.dropout(sublayer(self.norm(x))) # combining x with the output of previous layer called as sub layer and then applied dropout

# Now all above sub-blocks are combined as one block and there are "Nx" copies of these bigger blocks cascaded together. i.e. o/p of the previous one is connected with the input of the next one

# so we need to create a block which has 1x multi-head attention, 2x add & norm and 1x feedforward sub-blocks

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self._residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): # source mask applied to input of encoder to hide the interaction of padding words to other words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # first skip connection in diagram # selfattention is taking 3 same inputs
        x = self._residual_connections[1](x, self.feed_forward_block)
        return x
    

# now define encoder object. because encoder is made up of many encoder blocks.

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)















    
    










    



    








