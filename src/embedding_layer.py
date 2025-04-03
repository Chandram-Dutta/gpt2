import mlx.core as mx
import mlx.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim) # creates vectors for input tokens
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim) # adds positional information

    def __call__(self, input_tokens):

        batch_size, seq_len = input_tokens.shape

        positions = mx.arange(seq_len)[None, :]
        token_embeds = self.token_embedding(input_tokens)
        position_embeds = self.position_embedding(positions)

        return token_embeds + position_embeds
