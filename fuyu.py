import torch
import torch.nn as nn
from transformer import Transformer
from norms import LayerNorm
from utils import patch_img, threed_to_text_fixed
  
    
class Fuyu(nn.Module):
    def __init__(
        self,
        config,
        image_reshape_is_learnable: bool=False,
        patches: int=16,
    ):
        super().__init__()
        self.config = config
        self.patches = patches
        self.fuyu = Transformer(config)
        self.s_norm = LayerNorm(self.config.n_embd, bias=True)
        self.image_reshape_is_learnable = image_reshape_is_learnable
        
        if not image_reshape_is_learnable:
            # In case the image transformation does not need learnable parameters we will do it using fixed matrices
            random_img = torch.randn(1,3,256,256)
            random_img = patch_img(random_img, patches=self.patches)
            _, S, D = random_img.size()
            # Fixed transformation matrices.
            self.W1 = torch.randn(D, config.n_embd, device=config.device)
            self.W2 = torch.randn(S, config.block_size, device=config.device)
        else:
            # If the image reshaping needs to be learnable
            self.threed_to_text = nn.Sequential(
                nn.Linear(self.patches*self.patches*3, self.config.n_embd),
                nn.Linear(256, self.config.block_size)
            )
        
        print("The model has: %.2fM Parameters" % (self.get_num_params()/1e6,))
        
        
    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
        
    def forward(
        self, text: torch.Tensor, img: torch.Tensor=None, targets: torch.Tensor=None
    ):
        """
        Forward pass of the model.

        Text input shape: [batch, block_size, n_embd]
        img input shape: [batch, channels, height, width]

        Output shape: [batch, 1, vocab_size]

        """
        try:
            # If image is provided, concat it with the text
            if img is not None:
                # Patch the image
                img = patch_img(img, patches=self.patches)
                if self.image_reshape_is_learnable:
                    img = self.threed_to_text[0](img)
                    img = img.transpose(1, 2)
                    img = self.threed_to_text[1](img)
                    img = img.transpose(1, 2)
                else:
                    img = threed_to_text_fixed(img, self.W1, self.W2)
                img = self.s_norm(img)
            return self.fuyu(text, img, targets)
        except Exception as e:
            print(str(e))
            raise
    
    
    @torch.no_grad()
    def generate(self, text, img, max_new_tokens, temperature=1.0, top_k=None):
        if img is not None:
            # Patch the image
            img = patch_img(img, patches=self.patches)
            if self.image_reshape_is_learnable:
                img = self.threed_to_text[0](img)
                img = img.transpose(1, 2)
                img = self.threed_to_text[1](img)
                img = img.transpose(1, 2)
            else:
                img = threed_to_text_fixed(img, self.W1, self.W2)
            img = self.s_norm(img)
        return self.fuyu.generate(text, img, max_new_tokens, temperature, top_k)