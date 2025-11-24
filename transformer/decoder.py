import torch
import torch.nn as nn
import math


# ----------------------------
# 1. Positional Encoding (Sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# ----------------------------
# 2. Scaled Dot-Product Attention
# ----------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: [batch_size, n_heads, seq_len, d_k]
    mask: [batch_size, 1, 1, seq_len] or [seq_len, seq_len]
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, L, L]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # [B, H, L, d_k]
    return output, attn_weights


# ----------------------------
# 3. Multi-Head Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        Q = self.w_q(q)  # [B, L, D]
        K = self.w_k(k)
        V = self.w_v(v)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear
        return self.w_o(output)


# ----------------------------
# 4. Position-wise Feed-Forward Network
# ----------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()  # or ReLU

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ----------------------------
# 5. Decoder Layer
# ----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # for encoder-decoder
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        """
        x: [B, L_tgt, D] —— decoder input
        encoder_output: [B, L_src, D] —— optional, for encoder-decoder
        tgt_mask: causal mask for self-attention
        src_mask: padding mask for cross-attention
        """
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-Attention (if encoder_output provided)
        if encoder_output is not None:
            attn_out = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
            x = self.norm2(x + self.dropout(attn_out))

        # 3. Feed-Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# ----------------------------
# 6. Full Transformer Decoder
# ----------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tgt, encoder_output=None, src_mask=None, tgt_mask=None):
        """
        tgt: [B, L_tgt]
        encoder_output: [B, L_src, D] (optional)
        tgt_mask: [L_tgt, L_tgt] or [B, 1, L_tgt, L_tgt]
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # [B, L, D]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


# ----------------------------
# Helper: Create Causal Mask
# ----------------------------
def generate_square_subsequent_mask(sz):
    """Generate causal mask for decoder self-attention"""
    mask = torch.tril(torch.ones(sz, sz))  # [sz, sz]
    return mask  # 1s allowed, 0s masked


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Decoder-only (like GPT)
    decoder = TransformerDecoder(
        vocab_size=10000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_len=1024
    )

    tgt = torch.randint(0, 10000, (2, 10))  # [B=2, L=10]
    tgt_mask = generate_square_subsequent_mask(10)  # [10, 10]

    output = decoder(tgt, tgt_mask=tgt_mask)  # [2, 10, 512]
    print("Decoder-only output shape:", output.shape)

    # Encoder-Decoder (like T5)
    encoder_output = torch.randn(2, 15, 512)  # [B, L_src, D]
    src_mask = torch.ones(2, 1, 1, 15)  # [B, 1, 1, L_src]

    output2 = decoder(tgt, encoder_output, src_mask, tgt_mask)
    print("Encoder-Decoder output shape:", output2.shape)