"""
Temporal CKA Analysis Script

Analyzes how representations change during fine-tuning using Centered Kernel Alignment (CKA).
Loads epoch snapshots saved during training and computes:
1. CKA between consecutive epochs (how much does each layer change?)
2. CKA between null model (pretrained) and each finetuned epoch (what's preserved vs learned?)
3. CKA vs random initialization baseline (what's learned vs random weights)
4. Layer-wise CKA heatmaps

Terminology (per Mirco's guidance):
- Null model = Pretrained only (no finetuning) - the baseline for comparison
- Finetuned = Pretrained + finetuned on QA-SRL task
- Random = Randomly initialized (additional baseline showing pretraining effect)

Usage:
    python temporal_cka_analysis.py --model-dir /path/to/model/dir --config-name tiny

    # For Modal volume:
    modal run temporal_cka_analysis.py --model-dir TinyQA_spe200k_20251230-120000 --config-name tiny
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


# ============================================================
# Model Definitions (must match training script)
# ============================================================

class LLMConfig:
    def __init__(self, name: str, n_layers: int, hidden_size: int, n_heads: int, total_params_str: str = ""):
        self.name = name
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.total_params_str = total_params_str


class ToyMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attention_mask=None, is_causal=False):
        B, T, C = q.shape
        q_h = self.q_proj(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(v).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att_scores = (q_h @ k_h.transpose(-2, -1)) * (self.head_dim**-0.5)
        if is_causal:
            causal_mask = torch.triu(torch.ones_like(att_scores), diagonal=1).bool()
            att_scores = att_scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att_scores = att_scores.masked_fill(attention_mask == 0, float("-inf"))
        att_probs = torch.softmax(att_scores, dim=-1)
        att_probs = self.attn_dropout(att_probs)
        output = (att_probs @ v_h).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class ToyFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class ToyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ffn_expansion_factor: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = ToyMultiHeadAttention(hidden_size, n_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        ffn_hidden_size = hidden_size * ffn_expansion_factor
        self.ffn = ToyFeedForward(hidden_size, ffn_hidden_size, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None, is_causal=False):
        normed_x = self.norm1(x)
        attn_output = self.attention(normed_x, normed_x, normed_x, attention_mask=attention_mask, is_causal=is_causal)
        x = x + self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        return x


class ToyLLM(nn.Module):
    def __init__(self, config: LLMConfig, vocab_size: int, max_seq_len: int, pad_token_id: int,
                 ffn_expansion_factor: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_size)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([
            ToyTransformerBlock(config.hidden_size, config.n_heads, ffn_expansion_factor, dropout_rate)
            for _ in range(config.n_layers)
        ])
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None, is_causal=False, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.emb_dropout(tok_emb + pos_emb)

        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, is_causal=is_causal)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm_out(x)
        if output_hidden_states:
            hidden_states.append(x)
            return x, hidden_states
        return x


class ToyLLMForQuestionAnswering(nn.Module):
    def __init__(self, base_model: ToyLLM):
        super().__init__()
        self.llm = base_model
        self.qa_outputs = nn.Linear(self.llm.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        if output_hidden_states:
            sequence_output, hidden_states = self.llm(
                input_ids, attention_mask=attention_mask, is_causal=True, output_hidden_states=True
            )
            return sequence_output, hidden_states
        else:
            sequence_output = self.llm(input_ids, attention_mask=attention_mask, is_causal=True)
            return sequence_output


# ============================================================
# CKA Implementation
# ============================================================

def centering_matrix(n: int) -> np.ndarray:
    """Create centering matrix H = I - 1/n * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel K = X @ X^T"""
    return X @ X.T


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute Hilbert-Schmidt Independence Criterion"""
    n = K.shape[0]
    H = centering_matrix(n)
    return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two activation matrices.

    Args:
        X: (n_samples, features_x) activation matrix
        Y: (n_samples, features_y) activation matrix

    Returns:
        CKA similarity score in [0, 1]
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Compute kernels
    K = linear_kernel(X)
    L = linear_kernel(Y)

    # Compute CKA
    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    if hsic_kk < 1e-10 or hsic_ll < 1e-10:
        return 0.0

    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def cka_linear_fast(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Fast linear CKA implementation using the formula:
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    norm_XtX = np.linalg.norm(XtX, 'fro')
    norm_YtY = np.linalg.norm(YtY, 'fro')
    norm_YtX = np.linalg.norm(YtX, 'fro')

    if norm_XtX < 1e-10 or norm_YtY < 1e-10:
        return 0.0

    return (norm_YtX ** 2) / (norm_XtX * norm_YtY)


# ============================================================
# Data Loading
# ============================================================

class ProbeDataset(Dataset):
    """Simple dataset for extracting activations on a fixed probe set."""

    def __init__(self, tokenizer, max_seq_len: int, n_samples: int = 500):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load QA-SRL validation set
        print(f"Loading QA-SRL validation set for probe...")
        dataset = load_dataset("qa_srl", split="validation", trust_remote_code=True)

        self.examples = []
        for example in dataset:
            if len(self.examples) >= n_samples:
                break

            context = example["sentence"]
            question = " ".join([t for t in example["question"] if t != "_"])

            encoding = self.tokenizer(
                question, context,
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
                return_tensors="pt"
            )

            self.examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })

        print(f"Loaded {len(self.examples)} probe examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_probe(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return input_ids, attention_mask


# ============================================================
# Activation Extraction
# ============================================================

def extract_activations(model: ToyLLMForQuestionAnswering, dataloader: DataLoader,
                        device: torch.device) -> Dict[str, np.ndarray]:
    """
    Extract activations from each layer for the probe set.

    Returns:
        Dict mapping layer names to activation matrices (n_samples, hidden_size)
    """
    model.eval()

    all_activations = {f"layer_{i}": [] for i in range(model.llm.config.n_layers + 2)}
    # layer_0 = embeddings, layer_1..n = transformer blocks, layer_n+1 = final norm

    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, hidden_states = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Use last token's hidden state (or mean pool)
            for i, h in enumerate(hidden_states):
                # Mean pool over sequence length (ignoring padding)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
                all_activations[f"layer_{i}"].append(pooled.cpu().numpy())

    # Concatenate all batches
    return {k: np.concatenate(v, axis=0) for k, v in all_activations.items()}


# ============================================================
# Analysis Functions
# ============================================================

def compute_temporal_cka(activations_by_epoch: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Compute CKA between consecutive epochs for each layer.

    Returns:
        Dict mapping layer names to arrays of CKA scores between consecutive epochs
    """
    epochs = sorted(activations_by_epoch.keys())
    n_layers = len(list(activations_by_epoch.values())[0])
    layer_names = list(list(activations_by_epoch.values())[0].keys())

    results = {layer: [] for layer in layer_names}

    for i in range(len(epochs) - 1):
        epoch_a, epoch_b = epochs[i], epochs[i + 1]
        for layer in layer_names:
            cka = cka_linear_fast(
                activations_by_epoch[epoch_a][layer],
                activations_by_epoch[epoch_b][layer]
            )
            results[layer].append(cka)

    return {k: np.array(v) for k, v in results.items()}


def compute_cka_vs_pretrained(activations_pretrained: Dict[str, np.ndarray],
                               activations_by_epoch: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Compute CKA between pretrained model and each finetuning epoch.

    Returns:
        Dict mapping layer names to arrays of CKA scores vs pretrained
    """
    epochs = sorted(activations_by_epoch.keys())
    layer_names = list(activations_pretrained.keys())

    results = {layer: [] for layer in layer_names}

    for epoch in epochs:
        for layer in layer_names:
            cka = cka_linear_fast(
                activations_pretrained[layer],
                activations_by_epoch[epoch][layer]
            )
            results[layer].append(cka)

    return {k: np.array(v) for k, v in results.items()}


def compute_cka_vs_random(activations_random: Dict[str, np.ndarray],
                          activations_target: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute CKA between random (untrained) model and target model for each layer.

    This shows how different the target representations are from random initialization.

    Returns:
        Dict mapping layer names to CKA scores vs random model
    """
    layer_names = list(activations_random.keys())
    results = {}

    for layer in layer_names:
        cka = cka_linear_fast(activations_random[layer], activations_target[layer])
        results[layer] = cka

    return results


def compute_layer_cka_matrix(activations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute CKA between all pairs of layers for a single model.

    Returns:
        (n_layers, n_layers) CKA matrix
    """
    layer_names = sorted(activations.keys(), key=lambda x: int(x.split("_")[1]))
    n_layers = len(layer_names)

    matrix = np.zeros((n_layers, n_layers))
    for i, layer_i in enumerate(layer_names):
        for j, layer_j in enumerate(layer_names):
            if i <= j:
                cka = cka_linear_fast(activations[layer_i], activations[layer_j])
                matrix[i, j] = cka
                matrix[j, i] = cka

    return matrix


# ============================================================
# Visualization
# ============================================================

def plot_temporal_cka(cka_consecutive: Dict[str, np.ndarray],
                      cka_vs_pretrained: Dict[str, np.ndarray],
                      epochs: List[int],
                      output_path: str):
    """Plot temporal CKA analysis results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: CKA between consecutive epochs
    ax1 = axes[0]
    layer_names = sorted(cka_consecutive.keys(), key=lambda x: int(x.split("_")[1]))

    for layer in layer_names:
        ax1.plot(range(1, len(cka_consecutive[layer]) + 1), cka_consecutive[layer],
                 marker='o', label=layer, alpha=0.7)

    ax1.set_xlabel("Epoch Transition (t → t+1)")
    ax1.set_ylabel("CKA Similarity")
    ax1.set_title("Representation Stability Between Consecutive Epochs")
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Plot 2: CKA vs pretrained model
    ax2 = axes[1]
    for layer in layer_names:
        ax2.plot(epochs, cka_vs_pretrained[layer], marker='o', label=layer, alpha=0.7)

    ax2.set_xlabel("Finetuning Epoch")
    ax2.set_ylabel("CKA vs Pretrained")
    ax2.set_title("Representation Drift from Pretrained Model")
    ax2.legend(loc='lower left', fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved temporal CKA plot to {output_path}")
    plt.close()


def plot_random_baseline_comparison(cka_random_vs_pretrained: Optional[Dict[str, float]],
                                     cka_random_vs_finetuned: Dict[int, Dict[str, float]],
                                     epochs: List[int],
                                     output_path: str):
    """Plot CKA vs random (untrained) model baseline comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))
    layer_names = sorted(list(cka_random_vs_finetuned[epochs[0]].keys()), key=lambda x: int(x.split("_")[1]))

    # Plot CKA vs random for each epoch
    for layer in layer_names:
        cka_values = [cka_random_vs_finetuned[epoch][layer] for epoch in epochs]
        ax.plot(epochs, cka_values, marker='o', label=layer, alpha=0.7)

    # Add horizontal lines for pretrained vs random (if available)
    if cka_random_vs_pretrained is not None:
        colors = plt.cm.tab10.colors
        for i, layer in enumerate(layer_names):
            ax.axhline(y=cka_random_vs_pretrained[layer], color=colors[i % len(colors)],
                      linestyle='--', alpha=0.5)
        # Add legend note
        ax.plot([], [], 'k--', alpha=0.5, label='Pretrained vs Random')

    ax.set_xlabel("Finetuning Epoch")
    ax.set_ylabel("CKA vs Random (Untrained) Model")
    ax.set_title("Representation Divergence from Random Initialization")
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved random baseline plot to {output_path}")
    plt.close()


def plot_layer_cka_heatmap(cka_matrix: np.ndarray, title: str, output_path: str):
    """Plot layer-wise CKA heatmap."""

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1)

    n_layers = cka_matrix.shape[0]
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="CKA Similarity")

    # Add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            ax.text(j, i, f"{cka_matrix[i, j]:.2f}", ha="center", va="center",
                   color="white" if cka_matrix[i, j] < 0.5 else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved CKA heatmap to {output_path}")
    plt.close()


# ============================================================
# Main Analysis
# ============================================================

CONFIGS = {
    "tiny": {"n_layers": 2, "hidden_size": 128, "n_heads": 2},
    "small": {"n_layers": 4, "hidden_size": 256, "n_heads": 4},
    "base": {"n_layers": 6, "hidden_size": 512, "n_heads": 8},
    "medium": {"n_layers": 8, "hidden_size": 768, "n_heads": 12},
}


def main(model_dir: str, config_name: str, max_epochs: Optional[int] = None,
         n_probe_samples: int = 500, output_dir: Optional[str] = None):
    """
    Run temporal CKA analysis on saved epoch snapshots.

    Args:
        model_dir: Directory containing model checkpoints
        config_name: Model configuration name (tiny, small, base, medium)
        max_epochs: Maximum number of epochs to analyze (None = all)
        n_probe_samples: Number of samples for probe set
        output_dir: Directory for output plots (default: model_dir)
    """

    if output_dir is None:
        output_dir = model_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config_params = CONFIGS[config_name]
    config = LLMConfig(name=config_name, **config_params)
    print(f"Config: {config_name} ({config.n_layers} layers, {config.hidden_size} hidden)")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create model
    base_llm = ToyLLM(
        config=config,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        pad_token_id=tokenizer.pad_token_id,
        dropout_rate=0.0,  # Disable dropout for inference
    )
    model = ToyLLMForQuestionAnswering(base_llm).to(device)

    # Create probe dataset
    print("Creating probe dataset...")
    probe_dataset = ProbeDataset(tokenizer, max_seq_len=512, n_samples=n_probe_samples)
    probe_loader = DataLoader(probe_dataset, batch_size=32, shuffle=False, collate_fn=collate_probe)

    # Extract activations for random (untrained) model baseline
    print("Extracting random model activations (untrained baseline)...")
    activations_random = extract_activations(model, probe_loader, device)

    # Find epoch snapshots
    epoch_files = []
    for f in os.listdir(model_dir):
        if f.startswith("finetune_epoch_") and f.endswith(".pth"):
            epoch_num = int(f.replace("finetune_epoch_", "").replace(".pth", ""))
            epoch_files.append((epoch_num, os.path.join(model_dir, f)))

    epoch_files.sort(key=lambda x: x[0])

    if max_epochs:
        epoch_files = epoch_files[:max_epochs]

    print(f"Found {len(epoch_files)} epoch snapshots")

    if len(epoch_files) == 0:
        print("ERROR: No epoch snapshots found!")
        return

    # Extract activations for pretrained model
    pretrained_path = os.path.join(model_dir, "toy_llm_unified_pretrained.pth")
    activations_pretrained = None

    if os.path.exists(pretrained_path):
        print("Loading pretrained model...")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        # Handle the nested state dict from ToyLLMForPretraining
        llm_dict = {k.replace("llm.", ""): v for k, v in pretrained_dict.items() if k.startswith("llm.")}
        if llm_dict:
            model.llm.load_state_dict(llm_dict)
        else:
            model.llm.load_state_dict(pretrained_dict)

        print("Extracting pretrained activations...")
        activations_pretrained = extract_activations(model, probe_loader, device)
    else:
        print("WARNING: Pretrained model not found, skipping pretrained comparison")

    # Extract activations for each epoch
    activations_by_epoch = {}
    epochs = []

    for epoch_num, epoch_path in epoch_files:
        print(f"Processing epoch {epoch_num}...")

        checkpoint = torch.load(epoch_path, map_location=device)
        model.load_state_dict(checkpoint)

        activations_by_epoch[epoch_num] = extract_activations(model, probe_loader, device)
        epochs.append(epoch_num)

    # Compute CKA metrics
    print("\nComputing temporal CKA...")
    cka_consecutive = compute_temporal_cka(activations_by_epoch)

    cka_vs_pretrained = None
    if activations_pretrained is not None:
        print("Computing CKA vs pretrained...")
        cka_vs_pretrained = compute_cka_vs_pretrained(activations_pretrained, activations_by_epoch)

    # Compute CKA vs random (untrained) model baseline
    print("Computing CKA vs random model...")
    cka_random_vs_pretrained = None
    if activations_pretrained is not None:
        cka_random_vs_pretrained = compute_cka_vs_random(activations_random, activations_pretrained)

    # CKA between random and each finetuned epoch
    cka_random_vs_finetuned = {}
    for epoch_num in epochs:
        cka_random_vs_finetuned[epoch_num] = compute_cka_vs_random(activations_random, activations_by_epoch[epoch_num])

    # Save results
    results = {
        "epochs": epochs,
        "cka_consecutive": {k: v.tolist() for k, v in cka_consecutive.items()},
        "cka_random_vs_finetuned": {str(epoch): cka_random_vs_finetuned[epoch] for epoch in epochs},
    }
    if cka_vs_pretrained is not None:
        results["cka_vs_pretrained"] = {k: v.tolist() for k, v in cka_vs_pretrained.items()}
    if cka_random_vs_pretrained is not None:
        results["cka_random_vs_pretrained"] = cka_random_vs_pretrained

    results_path = os.path.join(output_dir, "temporal_cka_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Generate plots
    if cka_vs_pretrained is not None:
        plot_temporal_cka(
            cka_consecutive, cka_vs_pretrained, epochs,
            os.path.join(output_dir, "temporal_cka_plot.png")
        )

    # Layer CKA heatmaps for first and last epochs
    if len(epochs) >= 1:
        first_epoch = epochs[0]
        matrix_first = compute_layer_cka_matrix(activations_by_epoch[first_epoch])
        plot_layer_cka_heatmap(
            matrix_first, f"Layer CKA - Epoch {first_epoch}",
            os.path.join(output_dir, f"layer_cka_epoch_{first_epoch}.png")
        )

    if len(epochs) >= 2:
        last_epoch = epochs[-1]
        matrix_last = compute_layer_cka_matrix(activations_by_epoch[last_epoch])
        plot_layer_cka_heatmap(
            matrix_last, f"Layer CKA - Epoch {last_epoch}",
            os.path.join(output_dir, f"layer_cka_epoch_{last_epoch}.png")
        )

    # Plot random baseline comparison
    plot_random_baseline_comparison(
        cka_random_vs_pretrained, cka_random_vs_finetuned, epochs,
        os.path.join(output_dir, "random_baseline_cka_plot.png")
    )

    # Print summary
    print("\n" + "="*60)
    print("TEMPORAL CKA ANALYSIS SUMMARY")
    print("="*60)

    print("\nMean CKA between consecutive epochs (higher = more stable):")
    for layer in sorted(cka_consecutive.keys(), key=lambda x: int(x.split("_")[1])):
        mean_cka = np.mean(cka_consecutive[layer])
        print(f"  {layer}: {mean_cka:.4f}")

    if cka_vs_pretrained is not None:
        print("\nCKA vs null model (pretrained) at final epoch (higher = more preserved):")
        final_epoch = epochs[-1]
        for layer in sorted(cka_vs_pretrained.keys(), key=lambda x: int(x.split("_")[1])):
            final_cka = cka_vs_pretrained[layer][-1]
            print(f"  {layer}: {final_cka:.4f}")

    # Random baseline comparison
    if cka_random_vs_pretrained is not None:
        print("\nCKA: Random vs Pretrained (what pretraining learned from random init):")
        for layer in sorted(cka_random_vs_pretrained.keys(), key=lambda x: int(x.split("_")[1])):
            cka_val = cka_random_vs_pretrained[layer]
            print(f"  {layer}: {cka_val:.4f}")

    print("\nCKA: Random vs Final Finetuned (total change from random init):")
    final_epoch = epochs[-1]
    for layer in sorted(cka_random_vs_finetuned[final_epoch].keys(), key=lambda x: int(x.split("_")[1])):
        cka_val = cka_random_vs_finetuned[final_epoch][layer]
        print(f"  {layer}: {cka_val:.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal CKA Analysis")
    parser.add_argument("--model-dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--config-name", required=True, choices=["tiny", "small", "base", "medium"])
    parser.add_argument("--max-epochs", type=int, default=None, help="Max epochs to analyze")
    parser.add_argument("--n-probe-samples", type=int, default=500, help="Number of probe samples")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")

    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        config_name=args.config_name,
        max_epochs=args.max_epochs,
        n_probe_samples=args.n_probe_samples,
        output_dir=args.output_dir,
    )