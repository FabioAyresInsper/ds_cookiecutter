import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.transforms import ToTensor

vocab = torchtext.vocab.GloVe(name='6B', dim=50)
tokenizer = torchtext.data.get_tokenizer("basic_english")

class StringsToInt( nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def forward(self, data):
        return self.tokenizer.encode_as_ids(data)


class FixedSeq (nn.Module):
    def __init__(self, seq_len, pad_value):
        super().__init__()
        self.seq_len = seq_len
        self.pad_value = pad_value
    
    def forward(self, x):
        y = []
        for x_ in x:
            if len(x_) < self.seq_len:
                y_ = [self.pad_value] * (self.seq_len-len(x_)) + x_
            elif len(x_) > self.seq_len:
                #print(len(x_) - self.seq_len)
                idx = torch.randint(low=0, high=len(x_)-self.seq_len, size=(1,))[0].item()
                y_ = x_[idx:idx+self.seq_len]
            else:
                y_ = x_
            y.append(y_)
        return y   
                
class GetLast (nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x[:,:-1], x[:,1:]
    

class TextPredictorRNN ( nn.Module ):
    def __init__(self, vocabulary_size, n_latent, num_layers, embedding_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.encoding = nn.RNN(embedding_dim, n_latent, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.output = nn.Linear(n_latent,vocabulary_size)
    
    def forward(self, x):
        x = self.embedding_layer(x)
        enc, h_n = self.encoding(x)
        logits = self.output(enc)
        return logits, h_n

