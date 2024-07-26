import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


def l2norm(t, groups=1):
    B, H, T, E = t.size() # Batch, head size, block size, n_embd
    t = t.view(B, H, T, groups, E//groups) # we split the last dimension into groups
    t = F.normalize(t, dim=-1)
    return t.view(B, H, T, E) # reassemble the input


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def apply_rotary_pos_emb(t, freqs, scale=1):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)


def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (2, -1))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def patch_img(x: Tensor, patches: int):
    B, C, H, W = x.size()

    # Step 1: Reshape to (b, c, h//patch_size, patch_size, w//patch_size, patch_size)
    x_reshaped = x.view(B, C, H // patches, patches, W // patches, patches)

    # Step 2: Permute to (b, h//patch_size, w//patch_size, patch_size, patch_size, c)
    x_permuted = x_reshaped.permute(0, 2, 4, 3, 5, 1)

    # Step 3: Reshape to (b, h//patch_size * w//patch_size, patch_size * patch_size * c)
    x_final = x_permuted.contiguous().view(B, (H // patches) * (W // patches), patches * patches * C)
    return x_final


def threed_to_text_fixed(x: torch.Tensor, W1: Tensor, W2: Tensor):
    """
    Transforms a patched 3d image into a text representation using a fixed transformation provided by W1 and W2.
    x has to be patched.
    W1 has to have dimensions -> (x.size(-1), n_embd)
    W2 has to have dimensions -> (x.size(1), block_size)
    """
    x = x @ W1
    x = x.transpose(1, 2)
    x = x @ W2
    x = x.transpose(1, 2)
    
    return x