import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from norms import RMSNorm
from feedforward import MLP
from attention import CrossAttention, AlibiPositionalBias
from embeddings import RotaryEmbedding


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_2 = RMSNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = RMSNorm(config.n_embd)
        self.linear = nn.Linear(config.n_embd, config.n_embd)
        self.mlp = MLP(config)

        
    def forward(self, x, encoded_x, rotary_pos_emb, pos_emb):
        # We add x to each layer to skip connections.
        x = x + self.cross_attn(self.ln_2(x), encoded_x, rotary_pos_emb, pos_emb)
        x = x + self.mlp(self.ln_3(x))
        return x
    

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.base_std = 1/math.sqrt(config.n_embd) # The base std for initialization is calculated by taking the inverse of the sqrt of the embedding size
        
        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            rms = RMSNorm(config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            snorm = RMSNorm(config.n_embd),
        ))
        
        rotary_emb_dim = config.dim_head // 2
        self.rotary_pos_emb = RotaryEmbedding(
            rotary_emb_dim,
            scale_base=512,
            interpolation_factor=1,
            alpha=1,
        )
        self.pos_emb = AlibiPositionalBias(config.alibi_num_heads, config.n_head)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # This applies weight tying. It is useful to tie the weights of the head that generates the logit tokens with the token embedding layer.
        self.decoder.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=self.base_std/math.sqrt(2 * config.n_layer)) # Why 2? 2 operations on each Block

                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.base_std)

            
    def forward(self, idx, imgs, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.decoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        x = self.decoder.drop(self.decoder.rms(tok_emb))
        
        rotary_pos_emb = self.rotary_pos_emb(
            x.size(1), x.device
        )
        
        for idx in range(self.config.n_layer):
            x = self.decoder.h[idx](x, imgs, rotary_pos_emb, self.pos_emb)
            x = self.decoder.snorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # during inference we only need to apply the head to the temporal dimension
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    
    @torch.no_grad()
    def generate(self, idx, imgs, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            x = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(x, imgs)
                        
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            sample = torch.multinomial(probs, 1)

            idx = torch.cat((idx, sample), dim=-1)

        return idx