import os
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken

from transformer import Transformer

epochs = 5
batch_size = 10
seq_len = 1024
learning_rate = 1e-4
weight_decay = 0.01
n_layers = 4
hidden_dim = 256
max_seq_len = 256
n_heads = 4

enc = tiktoken.get_encoding("gpt2")

data_path = os.path.join(os.path.dirname(__file__), "test.txt")
try:
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Expected data file not found at {data_path}. Ensure that test.txt exists in the src directory.")

print(f"Loaded dataset with {len(text)} characters.")

tokens = enc.encode(text)
total_tokens = len(tokens)
print(f"Tokenized dataset contains {total_tokens} tokens.")

token_array = mx.array(tokens)

model = Transformer(n_layers=4, hidden_dim=256, max_seq_len=256, n_heads=4, vocab_size=enc.n_vocab)
optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
mx.eval(model.parameters())

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits[:, :-1, :], y[:, 1:], reduction="mean")

def get_batch(batch_size, seq_len):
    xs = []
    for _ in range(batch_size):
        start_idx = random.randint(0, total_tokens - seq_len - 1)
        segment = token_array[start_idx : start_idx + seq_len]
        xs.append(segment)
    x_batch = mx.stack(xs, axis=0)
    return x_batch, x_batch

dummy_seq_len = 128
dummy_batch, _ = get_batch(batch_size=1, seq_len=dummy_seq_len)
initial_loss = loss_fn(model, dummy_batch, dummy_batch)
mx.eval(initial_loss)
print(f"Initial Loss: {initial_loss.item():.3f}")

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
num_steps_per_epoch = total_tokens // (batch_size * seq_len)

print("Starting training...")
for epoch in range(epochs):
    print(f"--- Epoch {epoch+1}/{epochs} ---")
    for step in range(num_steps_per_epoch):
        x_batch, y_batch = get_batch(batch_size, seq_len)
        if x_batch is None or y_batch is None:
             print(f"Warning: Skipping step {step} due to missing batch data.")
             continue

        try:
            loss, grads = loss_and_grad_fn(model, x_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            print(f"Epoch {epoch+1} | Step {step:03d} | Loss: {loss.item():.3f}")
        except Exception as e:
            print(f"Error during training step {step}: {e}")
            break

print("Training finished.")

save_path = os.path.join(os.path.dirname(__file__), "..", "gpt2_model.safetensors")
print(f"Saving model parameters to {save_path}...")
model.save_weights(save_path)
print("Model saved.")
