import mlx.nn as nn
from transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, n_layers, hidden_dim, max_seq_len, n_heads, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hidden_dim, n_heads) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x
