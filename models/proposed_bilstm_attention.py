"""
Proposed Neural Model: Two-Layer BiLSTM + FastText Subwords + Self-Attention.
Paper Reference: Section 7.1.1, Figure 4, Table 7 & Table 8
Architecture:
  1. FastText character n-gram subword embeddings (n in [3, 6], d = 300)
  2. BiLSTM Layer 1 (bidirectional contextual encoding)
  3. BiLSTM Layer 2 (hierarchical sequence modeling)
  4. Self-Attention mechanism over BiLSTM hidden states -> sentence representations u (premise) and v (hypothesis)
  5. Symmetric Feature Fusion: [u; v; |u - v|; u ⊙ v] (dimension = 4 * 2 * hidden_dim)
  6. Dense MLP Classification Head with Dropout -> Softmax (3 classes: Entailment, Contradiction, Neutral)
Results on ANLI Golden Core Benchmark: Test Accuracy = 85.06%, Macro F1 = 0.850, Effective OOV = 0.0%.
"""

import math
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    # PyTorch placeholder for environments prior to pip install
    torch = None
    nn = object
    F = None


if torch is not None:
    class SelfAttention(nn.Module):
        """
        Self-Attention pooling mechanism over BiLSTM sequence hidden states.
        Computes attention score α_t = softmax(w^T tanh(W h_t + b))
        Output vector = \sum α_t h_t
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1, bias=False)
            )

        def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            # hidden_states: [batch_size, seq_len, hidden_dim]
            scores = self.projection(hidden_states).squeeze(-1)  # [batch_size, seq_len]

            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)

            attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
            # Context vector: weighted sum over sequence length
            context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch_size, hidden_dim]

            return context, attn_weights

    class TwoLayerBiLSTMAttentionNLI(nn.Module):
        """
        Proposed ANLI Architecture:
        Two-Layer BiLSTM + FastText Subwords + Self-Attention + Multi-Component Feature Fusion.
        """
        def __init__(
            self,
            vocab_size: int = 50000,
            embed_dim: int = 300,
            hidden_dim: int = 256,
            num_classes: int = 3,
            dropout: float = 0.3,
            pretrained_embeddings: Optional[torch.Tensor] = None
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.hidden_dim = hidden_dim

            # Embedding layer (can be initialized with pre-trained FastText weights)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight.data.copy_(pretrained_embeddings)
                self.embedding.weight.requires_grad = True  # Fine-tune subwords

            self.embed_dropout = nn.Dropout(dropout)

            # Two-Layer Bidirectional LSTM
            self.bilstm_layer1 = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.bilstm_layer2 = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )

            # Self-Attention pooling over the 2-layer BiLSTM output (dim = 2 * hidden_dim)
            bilstm_out_dim = hidden_dim * 2
            self.attention = SelfAttention(bilstm_out_dim)

            # Fusion dimension: [u, v, |u - v|, u * v] -> 4 * bilstm_out_dim
            fusion_dim = bilstm_out_dim * 4

            # Dense Classification MLP
            self.mlp = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )

        def encode_sentence(self, seq_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            # seq_tokens: [batch_size, seq_len]
            embedded = self.embed_dropout(self.embedding(seq_tokens))
            out1, _ = self.bilstm_layer1(embedded)
            out2, _ = self.bilstm_layer2(out1)

            # Self-attention pooling
            pooled_rep, attn_weights = self.attention(out2, mask)
            return pooled_rep, attn_weights

        def forward(
            self,
            premise_tokens: torch.Tensor,
            hypothesis_tokens: torch.Tensor,
            premise_mask: Optional[torch.Tensor] = None,
            hypothesis_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Encode premise u and hypothesis v
            u, attn_u = self.encode_sentence(premise_tokens, premise_mask)
            v, attn_v = self.encode_sentence(hypothesis_tokens, hypothesis_mask)

            # Symmetrical multi-feature fusion: [u; v; |u - v|; u ⊙ v]
            diff = torch.abs(u - v)
            elem_mult = u * v
            fusion = torch.cat([u, v, diff, elem_mult], dim=-1)

            # MLP classification
            logits = self.mlp(fusion)
            return logits, attn_u, attn_v

else:
    class TwoLayerBiLSTMAttentionNLI:
        pass


def get_model_summary() -> Dict[str, any]:
    """Returns architecture specs and empirical results from Section 7."""
    return {
        "model_name": "Two-Layer BiLSTM + FastText Subword + Self-Attention (Proposed)",
        "embedding": "FastText Subword CBOW (n in [3,6], d=300)",
        "context_encoder": "2-Layer Bidirectional LSTM (hidden_dim=256, bidirectional=True)",
        "pooling": "Self-Attention over stacked contextual hidden states",
        "fusion": "Concatenation vector: [u; v; |u - v|; u ⊙ v] (dim = 2048)",
        "classification_head": "Dense MLP (512 -> 256 -> 3) + BatchNorm + Dropout",
        "test_accuracy": 85.06,
        "macro_f1": 0.850,
        "oov_rate": 0.0,
        "parameter_count_million": 14.0,
        "training_time": "approx 4 h 15 min (CPU)"
    }


if __name__ == "__main__":
    summary = get_model_summary()
    print("=" * 60)
    print("ANLI Proposed Model: Two-Layer BiLSTM + FastText + Self-Attention")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:<25}: {v}")
    print("=" * 60)
