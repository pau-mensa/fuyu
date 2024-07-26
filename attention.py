import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import math
from utils import l2norm, pad_at_dim, apply_rotary_pos_emb, create_causal_mask


class AlibiPositionalBias(nn.Module):
    """
    Implementation of the Alibi positional bias. This will apply a negative bias to the sequence proportionally to its distance.
    This means that close tokens will have more relevance than further ones.
    """
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = slopes.unsqueeze(1).unsqueeze(2)
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            j_arange.unsqueeze(0).unsqueeze(0)
            - i_arange.unsqueeze(0).unsqueeze(2)
        )
        return bias

    
    @staticmethod
    def _get_slopes(heads):
        """"
        This function returns the slopes for the power of 2 of the heads. If the heads are not an exponent of 2 then we correct it.
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return sorted(
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ], reverse=True
        )

    
    @property
    def device(self):
        return next(self.buffers()).device

    
    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if (
            self.bias is not None
            and self.bias.shape[-1] >= j
            and self.bias.shape[-2] >= i
        ):
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.kv_heads = config.kv_heads

        q_dim = config.dim_head * config.n_head
        k_dim = config.dim_head * config.kv_heads
        v_dim = config.dim_head * config.kv_heads
        out_dim = config.dim_head * config.n_head
        
        self.qk_norm_q_scale = nn.Parameter(torch.ones(config.dim_head))
        self.qk_norm_k_scale = nn.Parameter(torch.ones(config.dim_head))
        
        # key, query, value projections for all heads, but in a batch
        self.linear_enc = nn.Linear(config.n_embd*2, config.n_embd, bias=config.bias)
        self.to_q = nn.Linear(config.n_embd, q_dim, bias=config.bias)
        self.to_k = nn.Linear(config.n_embd, k_dim, bias=config.bias)
        self.to_v = nn.Linear(config.n_embd, v_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(out_dim, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.kv_head = config.kv_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.mask = config.mask
        self.dim_head = config.dim_head
        self.qk_norm_scale = 10
        assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), "Flash Attention requires PyTorch >= 2.0"
        

    def forward(self, x, encoded_data, rotary_pos_emb, pos_emb):
        B, T, C = x.size() # batch size, block size, embedding dimensionality (dim_head)

        q = self.to_q(x) # batch size, sequence length, dim_head*n_head
        k = self.to_k(encoded_data) # batch size, sequence length, dim_head*kv_head
        v = self.to_v(encoded_data) # batch size, sequence length, dim_head*kv_head
        
        q = q.view(B, T, self.n_head, self.dim_head).transpose(1, 2)
        k = k.view(B, T, self.kv_head, self.dim_head).transpose(1, 2)
        v = v.view(B, T, self.kv_head, self.dim_head).transpose(1, 2)
        # We arrange the heads -> batch_size, n head (or kv head), block size, dim_head
        
        q, k = map(l2norm, (q, k))
        q = q * self.qk_norm_q_scale
        k = k * self.qk_norm_k_scale
        # We apply qk normalization on the dot products
        
        freqs, xpos_scale = rotary_pos_emb # block_size, dim_head/2
        l = freqs.shape[-1]
        
        q_xpos_scale, k_xpos_scale = (
            (xpos_scale, xpos_scale**-1.0)
            if xpos_scale is not None
            else (1.0, 1.0)
        )

        (ql, qr), (kl, kr), (vl, vr) = map(
            lambda t: (t[..., :l], t[..., l:]), (q, k, v)
        )
        # We split the qkv values into the left and right parts on the n_embd dimension.
        # batch, n_head, block_size, dim_head/2

        ql, kl, vl = map(
            lambda arg: apply_rotary_pos_emb(arg[0], freqs, arg[1]),
            ((ql, q_xpos_scale), (kl, k_xpos_scale), (vl, k_xpos_scale)),
        )
        # We apply the rotary embeddings only to the left part
        
        q, k, v = map(
            lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr))
        )
        # We rebuild the qkv values into the old dimensions -> batch_size, n head (or kv), block size, dim_head
        
        attn_bias = pos_emb(T, T)
        
        k = k.repeat(1,2,1,1)
        v = v.repeat(1,2,1,1)
        # We repeat the dimensions needed to the kv heads
        
        default_scale = q.shape[-1] ** -0.5
        q *= (default_scale / self.qk_norm_scale)
        # We apply normalization again to the q value?
            
        attn_bias = attn_bias.unsqueeze(0).expand(B, self.n_head, -1, -1)
        mask_value = -torch.finfo(q.dtype).max
        causal_mask = create_causal_mask(
            q.size(-2), k.size(-2), device=q.device
        )
        # We add the batch dimension to the attn bias

        attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)

        # attend:  (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.dropout * self.training, is_causal=False)  # [B, n_head, block_size, dim_head]
        y = y.transpose(1, 2).contiguous().view(B, T, self.dim_head * self.n_head) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y