from dataclasses import dataclass
import torch
from fuyu import Fuyu
import argparse

vocab_size = 10172
block_size = 2046
n_layer = 4
n_embd = 320
dropout = 0.0
n_head = 4
patches = 16

@dataclass
class DecoderConfig:
    block_size: int = 2046
    vocab_size: int = 10172
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 320
    dim_head: int = 64
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms. False: a bit better and faster
    mask: bool = False # Whether or not the attention is causal. (the future tokens get masked away)
    kv_heads: int = 2
    alibi_num_heads: int = 3
    device: str = 'cuda'
    

def random_data(with_targets: bool=False):
    """
    Function to generate random data for testing
    """
    # Text shape: [batch, block_size]
    text = torch.randint(0, vocab_size, (1, block_size), device=device)

    # Img shape: [batch, channels, height, width]
    img = torch.randn(1, 3, 256, 256, device=device)
    
    # Random targets to test the loss function
    targets = torch.randint(0, vocab_size, (1, block_size), device=device) if with_targets else None
        
    return text, img, targets


def test_forward_pass_without_targets(model):
    text, img, _ = random_data()
    # Apply model to text and img
    y = model(text, img)
    # Output shape: [batch, block_size, vocab_size] if targets are provided. [batch, 1, vocab_size] if targets are not provided
    print(y[0].size())
    
    
def test_forward_pass_with_targets(model):
    text, img, targets = random_data(with_targets=True)
    # Apply model to text and img
    y = model(text, img, targets=targets)
    # Output shape: [batch, block_size, vocab_size] if targets are provided. [batch, 1, vocab_size] if targets are not provided
    print(y[0].size(), y[1])
    
    
def test_generation(model, generated_sequence=30, temperature=1, top_k=200):
    text, img, _ = random_data()
    # Apply model to text and img
    model.eval()
    sample = model.generate(text, img, generated_sequence, temperature=temperature, top_k=top_k)
    print(sample.size())
    model.train()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Fuyu Test',
        description='This script tests the Fuyu model and architecture',
    )
    parser.add_argument('-fpt', '--forward_targets_pass',
                    action='store_true')
    parser.add_argument('-fp', '--forward_pass',
                    action='store_true')
    parser.add_argument('-g', '--generation',
                    action='store_true')
    
    args = parser.parse_args()
    
    print("Attempting initialization...")

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    bias = False
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    config = {k: globals()[k] for k in config_keys}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout, device=device) # init the model

    # This sets the matrix calculations precision to tensorfloat 32, which speeds up computation by a lot, with negligible cost for precision
    # Check https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html for more info
    torch.set_float32_matmul_precision('high')

    dconf = DecoderConfig(**model_args)
    model = Fuyu(
        dconf,
        patches=patches,
    )
    
    model.to(device)
    print("Initialization Done!")
    print("-----")
    
    test_all = not args.forward_targets_pass and not args.forward_pass and not args.generation
    
    if args.forward_pass or test_all:
        test_forward_pass_without_targets(model)
        print("Successful forward pass without targets!")
        print("-----")
        
    if args.forward_targets_pass or test_all:
        test_forward_pass_with_targets(model)
        print("Successful forward pass with targets!")
        print("-----")
        
    if args.generation or test_all:
        test_generation(model)
        print("Successful generation!")
        print("-----")

    print("Test successful!")