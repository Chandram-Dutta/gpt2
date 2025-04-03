import os
import mlx.core as mx
import tiktoken
from transformer import Transformer

n_layers = 4
hidden_dim = 256
max_seq_len = 256
n_heads = 4

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
print(f"Loaded tokenizer with vocabulary size: {vocab_size}")

model = Transformer(
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    max_seq_len=max_seq_len,
    n_heads=n_heads,
    vocab_size=vocab_size,
)

model_path = os.path.join(os.path.dirname(__file__), "..", "gpt2_model.safetensors")
print(f"Loading model from {model_path}...")

try:
    model.load_weights(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)


def generate_text(prompt, max_length=50, temperature=0.1):
    print(f"Generating text from prompt: {prompt}")

    tokens = mx.array(enc.encode(prompt))

    for _ in range(max_length):
        if len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:]

        x = tokens.reshape(1, -1)
        logits = model(x)
        next_token_logits = logits[0, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        probs = mx.softmax(next_token_logits)
        next_token = mx.random.categorical(probs, num_samples=1)[0]
        tokens = mx.concatenate([tokens, mx.array([next_token])])

    generated_text = enc.decode(tokens.tolist())
    return generated_text


if __name__ == "__main__":
    prompts = [
        "Valkyria Chronicles III is a tactical role",
        "The future of AI is",
        "In a world where technology",
        "The best way to learn programming is",
    ]

    for prompt in prompts:
        output = generate_text(prompt, max_length=200)
        print(f"\n=== Generated Text ===\n{output}\n")
        print("=" * 40)
