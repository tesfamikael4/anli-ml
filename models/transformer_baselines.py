"""
Transformer Fine-Tuning Baselines for Amharic NLI:
  1. Fine-Tuned mBERT (bert-base-multilingual-cased, WordPiece):
     Acc: 82.30%, Macro F1: 0.821, Effective OOV: 12.40% (Root Frag), 110M params
  2. Fine-Tuned XLM-RoBERTa-large (xlm-roberta-large, SentencePiece Unigram):
     Acc: 83.08%, Macro F1: 0.831, Effective OOV: 8.20%, 560M params
Paper Reference: Section 7.1 (Table 6, Table 7, Table 8)
"""

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    torch = None
    nn = object
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


class TransformerNLIBaseline:
    """
    Fine-tuning wrapper for HuggingFace multilingual transformers (mBERT and XLM-R)
    on the ANLI three-way classification task.
    """
    def __init__(self, model_checkpoint: str = "bert-base-multilingual-cased", num_labels: int = 3, device: str = "cpu"):
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint,
                num_labels=self.num_labels
            ).to(self.device)

    def prepare_inputs(self, premises: list, hypotheses: list, max_length: int = 128):
        if self.tokenizer is None:
            self.load_model()
        return self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)


def get_transformer_benchmark_results() -> Dict[str, Dict[str, any]]:
    return {
        "mBERT": {
            "model_name": "Fine-Tuned mBERT (WordPiece)",
            "checkpoint": "bert-base-multilingual-cased",
            "accuracy": 82.30,
            "macro_f1": 0.821,
            "effective_oov_root_frag": 12.40,
            "parameters_m": 110.0,
            "learning_rate": "2e-5",
            "optimizer": "AdamW",
            "batch_size": 16,
            "epochs": 5
        },
        "XLM_RoBERTa_large": {
            "model_name": "Fine-Tuned XLM-RoBERTa-large",
            "checkpoint": "xlm-roberta-large",
            "accuracy": 83.08,
            "macro_f1": 0.831,
            "effective_oov_rate": 8.20,
            "parameters_m": 560.0,
            "learning_rate": "2e-5",
            "optimizer": "AdamW",
            "batch_size": 8,
            "epochs": 5
        }
    }
