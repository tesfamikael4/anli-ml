# amharic_nli
**A Multi-Domain Benchmark Corpus, Automated Quality Filtering, and Neural Models for Amharic Natural Language Inference**

Part of the research project:
> **"ANLI: A Multi-Domain Benchmark Corpus, Automated Quality Filtering, and Subword Methodologies for Amharic Natural Language Inference"**
> *Tesfaye Bekele Kassa, Dr. Tesfa Tegegne Asfaw, Beyene Feleke Mekonnen*
> *Woldia University & Bahir Dar University, Ethiopia*

---

## 🌟 Overview

This repository contains the **supervised model architectures**, **ANLI Golden Core benchmark (210,602 pairs across 11 communicative domains)**, and **training/evaluation pipelines**.

### Key Architectural Results
- **Proposed Architecture (`train_proposed_model.py`, `models/proposed_bilstm_attention.py`)**:
  - Two-Layer BiLSTM + FastText Subwords ($d=300$) + Additive Self-Attention Pooling + Symmetric Interactive Matching $[u; v; |u-v|; u \odot v]$
  - **State-of-the-Art Test Accuracy**: **$85.06\%$** (Macro F1 = $0.850$)
  - **Extreme Parameter Efficiency**: Only **$14.8\text{M}$ parameters** (outperforming 560M-parameter XLM-RoBERTa-large and AfroXLMR with $37.8\times$ fewer parameters)
- **Ablation & Statistical Validation**:
  - `table2_geez_normalization_ablation.py`: Evaluates the 4 Ge'ez normalization paradigms (Table 2)
  - `table6_embedding_tokenization_ablation.py`: Evaluates the 7 embedding and tokenization schemes (Table 6)
  - `table7_table8_anli_benchmark.py`: Complete 3-class results matrix & Table 8 training configurations
  - `evaluate_baselines.py`: McNemar's paired test ($\chi^2 \ge 18.94, p < 0.0001$ against all baselines)

---

## 📂 Repository Structure

```
amharic_nli/
├── README.md
├── requirements.txt
├── dataset_and_tokenization.py     # ANLI PyTorch Dataset & Subword Tokenizer (loads all_in_one_cleaned.jsonl)
├── train_proposed_model.py         # Full training pipeline with AdamW & Cosine Annealing
├── evaluate_baselines.py           # Table 7 benchmark runner & McNemar significance tests
├── inference_and_diagnostics.py    # Interactive premise-hypothesis reasoning & error taxonomy
├── all_in_one_cleaned.jsonl        # Cleaned ANLI benchmark dataset with 5 attributes
└── models/
    ├── proposed_bilstm_attention.py # Two-Layer BiLSTM + FastText + Self-Attention (Figure 4)
    ├── bilstm_baselines.py          # Single BiLSTM (Word2Vec / FastText)
    └── transformer_baselines.py     # mBERT, XLM-RoBERTa, and AfroXLMR wrappers
```

### Dataset Schema (`all_in_one_cleaned.jsonl`)
Each line in `all_in_one_cleaned.jsonl` is a JSON object formatted as:
```json
{
  "premise": "ጠቅላይ ሚኒስትሩ አዲስ የኢኮኖሚ ማሻሻያ አዋጅ ይፋ አደረጉ።",
  "hypothesis": "የሀገሪቱ የኢኮኖሚ ፖሊሲ ላይ ለውጥ ተካሂዷል።",
  "label": "entailment",
  "sub-domain": "Politics_and_Governance",
  "domain": "News_and_Media"
}
```
- **`premise`**: Reference context sentence in normalized Amharic
- **`hypothesis`**: Proposition statement to infer
- **`label`**: 3-Way Gold Label (`entailment`, `neutral`, `contradiction`)
- **`sub-domain`**: Specialized topic/genre (e.g. `Politics_and_Governance`, `Public_Forums`, `Clinical_Medicine`, `Judicial_Courts`, `Macroeconomics_and_Banking`, etc.)
- **`domain`**: One of the 11 communicative domains

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Proposed Two-Layer BiLSTM + Self-Attention Architecture
```bash
python train_proposed_model.py \
    --data all_in_one_cleaned.jsonl \
    --epochs 20 \
    --batch_size 64 \
    --lr 7.5e-4
```

### 3. Evaluate Against All Baselines & McNemar Paired Tests (Table 7)
```bash
python evaluate_baselines.py
```

### 4. Run Interactive Inference & Diagnostic Analysis
```bash
python inference_and_diagnostics.py
```
