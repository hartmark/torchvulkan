import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each head's two halves by its position's angle (rotary position embedding)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Attention(nn.Module):
    """Causal grouped-query attention: ``n_kv_head`` key/value heads are shared by ``n_head`` query heads."""

    def __init__(self, n_embd: int, n_head: int, n_kv_head: int):
        super().__init__()
        self.n_head, self.n_kv_head = n_head, n_kv_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.kv = nn.Linear(n_embd, 2 * n_kv_head * self.head_dim, bias=False)
        self.out = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.n_kv_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(n_embd, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, n_kv_head)
        self.mlp_norm = nn.RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, 64 * round(8 * n_embd / 3 / 64))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """A small modern decoder-only transformer: pre-RMSNorm, rotary embeddings, grouped-query
    attention, a SwiGLU MLP and tied input/output embeddings.

    ``n_embd`` sets the model width; ``benchmark/benchmark.py llm`` sweeps it to scale the model up,
    holding ``n_layer`` fixed and deriving the head counts from a constant per-head size.
    """

    def __init__(self, vocab_size: int = 256, block_size: int = 128, n_layer: int = 4, n_embd: int = 256, head_dim: int = 64, kv_groups: int = 2):
        super().__init__()
        assert n_embd % head_dim == 0
        n_head = n_embd // head_dim
        n_kv_head = n_head // kv_groups if n_head % kv_groups == 0 else n_head
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_kv_head) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.tok_emb.weight, std=0.02)  # the shared head would start with huge logits at nn.Embedding's N(0, 1)
        cos, sin = rope_cache(block_size, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        T = idx.size(1)
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, steps: int, temperature: float = 0.8):
        for _ in range(steps):
            logits = self(idx[:, -self.block_size:])[:, -1] / temperature
            idx = torch.cat((idx, torch.multinomial(logits.softmax(-1), 1)), dim=1)
        return idx


def main():
    import torchvulkan

    block_size = 128
    batch_size = 32
    steps = 300
    device = torch.device("vulkan" if torchvulkan.is_available() else "cpu")

    # a character-level model trained on this file's own source, so no dataset download is needed
    data = torch.tensor(list(open(__file__, "rb").read()), dtype=torch.long)

    def make_batch():
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        windows = torch.stack([data[s:s + block_size + 1] for s in starts.tolist()])
        return windows[:, :-1].to(device), windows[:, 1:].to(device)

    model = GPT(block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    model.train()
    for step in range(steps):
        inputs, targets = make_batch()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"\rstep {step + 1}/{steps} | loss {loss.item():.4f}", end="")

    model.eval()
    prompt = torch.tensor([list(b"def ")], device=device)
    sample = model.generate(prompt, 200)[0].tolist()
    print("\n\n" + bytes(sample).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
