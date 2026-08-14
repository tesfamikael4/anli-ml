"""
Baseline Models for ANLI Benchmark:
  1. Single BiLSTM + Word2Vec Baseline (Acc: 74.80%, Macro F1: 0.747, OOV: 34.20%)
  2. Single BiLSTM + FastText Subword Baseline (Acc: 83.52%, Macro F1: 0.835, OOV: 0.0%)
Paper Reference: Section 7.1 (Table 6, Table 7, Table 8)
"""

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object
    F = None


if torch is not None:
    class SingleBiLSTMNLI(nn.Module):
        """
        Single-layer BiLSTM baseline with Max/Mean Pooling for sentence encoding.
        Supports atomic Word2Vec lookup (with OOV token) or FastText Subword embeddings.
        """
        def __init__(
            self,
            vocab_size: int = 50000,
            embed_dim: int = 300,
            hidden_dim: int = 256,
            num_classes: int = 3,
            dropout: float = 0.3,
            use_fasttext: bool = True
        ):
            super().__init__()
            self.use_fasttext = use_fasttext
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)

            self.bilstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )

            # Symmetrical concatenation fusion: [u, v, |u - v|, u * v]
            fusion_dim = (hidden_dim * 2) * 4

            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )

        def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            emb = self.dropout(self.embedding(x))
            out, (hn, cn) = self.bilstm(emb)
            # Max-pooling across sequence dimension
            if mask is not None:
                out = out.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled, _ = torch.max(out, dim=1)
            return pooled

        def forward(
            self,
            premise: torch.Tensor,
            hypothesis: torch.Tensor,
            premise_mask: Optional[torch.Tensor] = None,
            hypothesis_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            u = self.encode(premise, premise_mask)
            v = self.encode(hypothesis, hypothesis_mask)

            fusion = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
            logits = self.classifier(fusion)
            return logits

else:
    class SingleBiLSTMNLI:
        pass


def get_baseline_metrics() -> Dict[str, Dict[str, any]]:
    """Returns baseline empirical metrics from the paper."""
    return {
        "bilstm_word2vec": {
            "name": "Single BiLSTM + Word2Vec",
            "accuracy": 74.80,
            "macro_f1": 0.747,
            "oov_rate": 34.20,
            "parameters_m": 12.0,
            "training_time": "~3 h 20 min"
        },
        "bilstm_fasttext": {
            "name": "Single BiLSTM + FastText Subword (Ours)",
            "accuracy": 83.52,
            "macro_f1": 0.835,
            "oov_rate": 0.00,
            "parameters_m": 12.5,
            "training_time": "~3 h 45 min"
        }
    }
