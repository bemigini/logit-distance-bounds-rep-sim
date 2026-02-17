import torch.nn
from torch import nn

from backbones.base.ops import Flatten

class MLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_units=512, 
                 repr_dim=2, 
                 dropout=0.1):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.dropout = nn.Dropout(p=dropout)

        self.enc_block_1 = nn.Linear(input_dim, hidden_units)
        self.enc_block_2 = nn.Linear(hidden_units, hidden_units)
        self.enc_block_3 = nn.Linear(hidden_units, hidden_units)
        self.enc_final = nn.Linear(hidden_units, repr_dim)

    def forward(self, x):
        # BLOCK 1
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        x = self.enc_block_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # BLOCK 2
        x = self.enc_block_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # BLOCK3
        x = self.enc_block_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # FINAL MAP TO REPRS
        x = self.enc_final(x)
        return x
    
class Linear(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_units=512, 
                 repr_dim=2, 
                 dropout=0.1):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim

        self.flatten = Flatten()
        
        self.enc_block_1 = nn.Linear(input_dim, repr_dim)

    def forward(self, x):
        # BLOCK 1
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        x = self.enc_block_1(x)
        return x
    

class Shallow(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_units=512, 
                 repr_dim=2, 
                 dropout=0.1):
        super(Shallow, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.dropout = nn.Dropout(p=dropout)

        self.enc_block_1 = nn.Linear(input_dim, hidden_units)
        self.enc_final = nn.Linear(hidden_units, repr_dim)

    def forward(self, x):
        # BLOCK 1
        x = self.flatten(x)  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        x = self.enc_block_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # FINAL MAP TO REPRS
        x = self.enc_final(x)
        return x