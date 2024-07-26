import torch
import torch.nn as nn
from typing import Callable
from norms import LayerNorm


class GLU(nn.Module):
    
    def __init__(self, dim_in, dim_out, activation: Callable, mult_bias=False):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

        
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias
    
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.glu     = GLU(config.n_embd, 4 * config.n_embd, nn.SiLU())
        self.norm    = LayerNorm(4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        
    def forward(self, x):
        x = self.glu(x)
        x = self.c_proj(self.norm(x))
        x = self.dropout(x)
        return x