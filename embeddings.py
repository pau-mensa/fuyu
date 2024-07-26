import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=True,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        alpha=1.0,
    ):
        super().__init__()
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= alpha ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        t = t / self.interpolation_factor
        
        freqs = t[:, None] * self.inv_freq[None, :]
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.0

        power = (
            torch.arange(seq_len, device=device) - (seq_len // 2)
        ) / self.scale_base
        scale = self.scale ** power.unsqueeze(1)
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale