import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.layerNorm1 = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiHeadAttention(dims=hidden_dim, num_heads=n_heads, bias=True)
        self.layerNorm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def __call__(self, x):
        seq_len = x.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        x_norm = self.layerNorm1(x)
        attn_output = self.mha(x_norm, x_norm, x_norm, mask=causal_mask)
        x = x + attn_output

        x_norm = self.layerNorm2(x)
        mlp = self.linear1(x_norm)
        mlp = self.gelu(mlp)
        mlp = self.linear2(mlp)
        x = x + mlp

        return x
