"""
Modal version of pretrain_wikitext_finetune_qasrl.py

This script runs the ToyLLM pre-training and fine-tuning pipeline on Modal's cloud GPUs.

Usage:
    # Run with default settings (tiny config)
    modal run pretrain_wikitext_finetune_qasrl_modal.py

    # Run with specific config
    modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small

    # Run with custom GPU
    modal run pretrain_wikitext_finetune_qasrl_modal.py --gpu a100

    # Deploy as a persistent app
    modal deploy pretrain_wikitext_finetune_qasrl_modal.py
"""

import modal
import sys

# Parse config name from command line to create unique app name
_config_name = "tiny"  # default
for i, arg in enumerate(sys.argv):
    if arg == "--config-name" and i + 1 < len(sys.argv):
        _config_name = sys.argv[i + 1]
        break

# Define the Modal app with config-specific name
app = modal.App(f"llm-pretrain-{_config_name}")

# Create persistent volumes for data and models
data_volume = modal.Volume.from_name("llm-training-data", create_if_missing=True)
models_volume = modal.Volume.from_name("llm-models", create_if_missing=True)

# Pre-download datasets during image build to avoid GPU idle time
def download_datasets():
    from datasets import load_dataset
    # QA-SRL requires trust_remote_code due to custom loading script
    load_dataset("qa_srl", split="train", trust_remote_code=True)
    load_dataset("qa_srl", split="validation", trust_remote_code=True)
    load_dataset("squad", split="train")
    load_dataset("squad", split="validation")
    load_dataset("nq_open", split="train")
    load_dataset("nq_open", split="validation")
    load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")


# Define the image with all required dependencies
# Note: datasets<3.0.0 required because qa_srl uses a loading script
# which is no longer supported in datasets 3.0+
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0,<3.0.0",  # Pin to <3.0 for qa_srl loading script support
        "accelerate>=0.21.0",
    )
    .run_function(download_datasets)
)

# Configuration definitions (embedded to avoid file dependencies)
# Note: All models use identical pretraining stopping criteria (200 epochs, patience=20)
# to ensure fair comparison across scales. Models will converge at different epochs.
#
# GPU recommendations:
#   - tiny, small, base: A10G (default)
#   - medium: A100-80GB (for batch=96 without OOM)
CONFIGS = {
    "tiny": {
        "llm_config": {
            "name": "TinyQA",
            "n_layers": 2,
            "hidden_size": 128,
            "n_heads": 2
        },
        "training_params": {
            "max_seq_len": 512,
            "dropout_rate": 0.05,  # Reduced from 0.15 - tiny models underfit with high dropout
            "pretrain_lr": 5e-4,  # Slightly reduced for stability
            "num_pretrain_epochs": 200,  # Generous cap, let early stopping decide
            "pretrain_patience": 20,  # Consistent across all scales
            "warmup_steps": 500,
            "batch_size_pretrain": 64,  # A10G - near max due to vocab size (30K) output logits
            "gradient_accumulation_steps": 1,
            "samples_per_epoch": 200_000,  # ~12 min epochs, appropriate for tiny model capacity
            "num_finetune_epochs": 30,
            "finetune_lr": 4e-5,
            "finetune_patience": 8,
            "finetune_warmup_steps": 200,
            "batch_size_qa": 64,
            "recommended_gpu": "A10G",
            "use_squad_finetune": False  # Use only QA-SRL for clean emergence testing
        }
    },
    "small": {
        "llm_config": {
            "name": "SmallQA",
            "n_layers": 4,
            "hidden_size": 256,
            "n_heads": 4
        },
        "training_params": {
            "max_seq_len": 512,
            "dropout_rate": 0.1,
            "pretrain_lr": 2e-4,  # Lowered from 3e-4 for stability with ~2M params
            "num_pretrain_epochs": 200,  # Generous cap, let early stopping decide
            "pretrain_patience": 20,  # Consistent across all scales
            "warmup_steps": 500,
            "batch_size_pretrain": 48,  # A10G optimized - no grad accumulation needed
            "gradient_accumulation_steps": 1,  # Faster than accumulating
            "samples_per_epoch": 400_000,  # ~25 min epochs, scaled for small model capacity (~2M params)
            "num_finetune_epochs": 30,
            "finetune_lr": 2e-5,
            "finetune_patience": 8,
            "finetune_warmup_steps": 300,
            "batch_size_qa": 48,
            "recommended_gpu": "A10G",
            "use_squad_finetune": False  # Use only QA-SRL for clean emergence testing
        }
    },
    "base": {
        "llm_config": {
            "name": "BaseQA",
            "n_layers": 6,
            "hidden_size": 512,
            "n_heads": 8
        },
        "training_params": {
            "max_seq_len": 512,
            "dropout_rate": 0.1,
            "pretrain_lr": 2e-4,  # Adjusted for batch=32 (was 3e-4 for effective batch=48)
            "num_pretrain_epochs": 200,  # Generous cap, let early stopping decide
            "pretrain_patience": 20,  # Consistent across all scales
            "warmup_steps": 1000,
            "batch_size_pretrain": 32,  # A10G - no grad accumulation for faster training
            "gradient_accumulation_steps": 1,  # Faster than accumulating
            "samples_per_epoch": 800_000,  # ~40 min epochs, scaled for base model capacity (~50M params)
            "num_finetune_epochs": 30,
            "finetune_lr": 1e-5,
            "finetune_patience": 8,
            "finetune_warmup_steps": 500,
            "batch_size_qa": 32,
            "recommended_gpu": "A10G",
            "use_squad_finetune": False  # Use only QA-SRL for clean emergence testing
        }
    },
    "medium": {
        "llm_config": {
            "name": "MediumQA",
            "n_layers": 8,
            "hidden_size": 768,
            "n_heads": 12
        },
        "training_params": {
            "max_seq_len": 512,
            "dropout_rate": 0.1,
            "pretrain_lr": 1e-4,  # Increased for larger batch (linear scaling rule)
            "num_pretrain_epochs": 200,  # Generous cap, let early stopping decide
            "pretrain_patience": 20,  # Consistent across all scales
            "warmup_steps": 2000,
            "batch_size_pretrain": 96,  # A100-80GB optimized - double throughput
            "gradient_accumulation_steps": 1,  # Faster on A100 with larger batch
            "samples_per_epoch": 1_500_000,  # Near full dataset, scaled for medium model capacity (~100M+ params)
            "num_finetune_epochs": 40,
            "finetune_lr": 3e-6,
            "finetune_patience": 10,
            "finetune_warmup_steps": 1000,
            "batch_size_qa": 96,
            "recommended_gpu": "A100-80GB",  # Required for batch=96 without OOM
            "use_squad_finetune": False  # Use only QA-SRL for clean emergence testing
        }
    }
}


# Training function factory - the actual implementation
def _train_impl(
    config_name: str = "tiny",
    skip_pretrain: bool = False,
    skip_finetune: bool = False,
    resume_path: str = None,
    checkpoint_interval: int = 1,
    freeze_llm: bool = False,
    finetune_samples: int = None,
    pretrain_data: str = "wiki",
    random_baseline: bool = False,
    _continuation_count: int = 0,
    train_fn=None,  # Modal function reference for spawning continuations
):
    """
    Main training function that runs on Modal GPU.

    Args:
        config_name: Configuration to use ('tiny', 'small', 'base', 'medium')
        skip_pretrain: Skip the pre-training phase
        skip_finetune: Skip the fine-tuning phase
        resume_path: Path to checkpoint to resume from (relative to /models)
        checkpoint_interval: Save checkpoint every N epochs
        freeze_llm: Freeze LLM weights during finetuning (linear probe mode)
        finetune_samples: Limit finetuning to N samples (for few-shot experiments)
        pretrain_data: Pretraining data to use ('wiki' or 'wiki+nq')
        random_baseline: Use randomly initialized LLM (no pretrained weights) as baseline
    """
    import json
    import math
    import os
    import time
    import random
    from typing import List

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from datasets import load_dataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
    from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset
    from transformers import AutoTokenizer

    # ============================================================
    # Timeout and Auto-Continuation Settings
    # ============================================================
    JOB_START_TIME = time.time()
    MAX_RUNTIME_SECONDS = 23 * 3600  # 23 hours (1 hour buffer before 24h limit)
    TIMEOUT_CHECK_INTERVAL = 100  # Check timeout every N batches

    def is_approaching_timeout():
        """Check if we're approaching the Modal timeout limit."""
        elapsed = time.time() - JOB_START_TIME
        return elapsed > MAX_RUNTIME_SECONDS

    def get_elapsed_time_str():
        """Get human-readable elapsed time."""
        elapsed = time.time() - JOB_START_TIME
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"

    # Set environment variables for HuggingFace
    HF_CACHE_DIR = "/data/huggingface_cache"
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

    print(f"Starting training with config: {config_name}")
    print(f"Skip pretrain: {skip_pretrain}, Skip finetune: {skip_finetune}")
    print(f"Continuation count: {_continuation_count}")
    print(f"Max runtime: {MAX_RUNTIME_SECONDS / 3600:.1f} hours")

    # Check GPU recommendation
    recommended_gpu = CONFIGS[config_name]["training_params"].get("recommended_gpu", "A10G")
    if torch.cuda.is_available():
        current_gpu = torch.cuda.get_device_name(0)
        if recommended_gpu == "A100" and "A100" not in current_gpu:
            print(f"\n{'!'*60}")
            print(f"WARNING: Config '{config_name}' recommends A100 GPU for batch_size={CONFIGS[config_name]['training_params']['batch_size_pretrain']}")
            print(f"Current GPU: {current_gpu}")
            print(f"You may experience OOM errors. To use A100:")
            print(f"  1. Change @app.function decorator: gpu='A100'")
            print(f"  2. Or reduce batch_size in config")
            print(f"{'!'*60}\n")

    # ============================================================
    # Model Definitions
    # ============================================================

    class LLMConfig:
        """Holds the configuration for a toy LLM."""

        def __init__(
            self,
            name: str,
            n_layers: int,
            hidden_size: int,
            n_heads: int,
            total_params_str: str = "",
        ):
            self.name = name
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.n_heads = n_heads
            self.total_params_str = total_params_str
            if hidden_size % n_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
                )

    class ToyMultiHeadAttention(nn.Module):
        """Multi-Head Attention module for the ToyLLM."""

        def __init__(self, hidden_size: int, n_heads: int, dropout_rate: float = 0.1):
            super().__init__()
            assert hidden_size % n_heads == 0
            self.hidden_size = hidden_size
            self.n_heads = n_heads
            self.head_dim = hidden_size // n_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.attn_dropout = nn.Dropout(dropout_rate)  # Attention probability dropout

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
            att_probs = self.attn_dropout(att_probs)  # Apply attention dropout
            output = (att_probs @ v_h).transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(output)

    class ToyFeedForward(nn.Module):
        """Feed-forward network module."""

        def __init__(
            self, hidden_size: int, ffn_hidden_size: int, dropout_rate: float = 0.1
        ):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
            self.activation = nn.GELU()
            self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            return self.dropout(self.linear2(self.activation(self.linear1(x))))

    class ToyTransformerBlock(nn.Module):
        """A single Transformer block combining attention and a feed-forward network."""

        def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            ffn_expansion_factor: int = 4,
            dropout_rate: float = 0.1,
        ):
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
            attn_output = self.attention(
                normed_x,
                normed_x,
                normed_x,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
            x = x + self.dropout1(attn_output)
            ffn_output = self.ffn(self.norm2(x))
            x = x + self.dropout2(ffn_output)
            return x

    class ToyLLM(nn.Module):
        """Base ToyLLM, without a task-specific head."""

        def __init__(
            self,
            config: LLMConfig,
            vocab_size: int,
            max_seq_len: int,
            pad_token_id: int,
            ffn_expansion_factor: int = 4,
            dropout_rate: float = 0.1,
        ):
            super().__init__()
            self.config = config
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len
            self.pad_token_id = pad_token_id

            self.token_embedding = nn.Embedding(
                vocab_size, config.hidden_size, padding_idx=pad_token_id
            )
            self.position_embedding = nn.Embedding(max_seq_len, config.hidden_size)
            self.emb_dropout = nn.Dropout(dropout_rate)
            self.layers = nn.ModuleList(
                [
                    ToyTransformerBlock(
                        config.hidden_size,
                        config.n_heads,
                        ffn_expansion_factor,
                        dropout_rate,
                    )
                    for _ in range(config.n_layers)
                ]
            )
            self.norm_out = nn.LayerNorm(config.hidden_size)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

        def forward(self, input_ids, attention_mask=None, is_causal=False):
            batch_size, seq_len = input_ids.shape
            if seq_len > self.max_seq_len:
                input_ids = input_ids[:, : self.max_seq_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, : self.max_seq_len]
                seq_len = self.max_seq_len

            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

            tok_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(position_ids)

            x = self.emb_dropout(tok_emb + pos_emb)

            for layer in self.layers:
                x = layer(x, attention_mask=attention_mask, is_causal=is_causal)

            sequence_output = self.norm_out(x)
            return sequence_output

    class ToyLLMForQuestionAnswering(nn.Module):
        """ToyLLM with a QA head for fine-tuning."""

        def __init__(self, base_model: ToyLLM):
            super().__init__()
            self.llm = base_model
            self.qa_outputs = nn.Linear(self.llm.config.hidden_size, 2)

            # Initialize QA head with same distribution as LLM layers for consistency
            torch.nn.init.normal_(self.qa_outputs.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.qa_outputs.bias)

        def forward(
            self,
            input_ids,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            context_mask=None,
        ):
            sequence_output = self.llm(
                input_ids, attention_mask=attention_mask, is_causal=True
            )
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # Apply context mask to ensure loss only considers valid answer positions
                # This matches evaluation behavior where predictions are restricted to context
                if context_mask is not None:
                    start_logits_for_loss = start_logits.masked_fill(context_mask == 0, float('-inf'))
                    end_logits_for_loss = end_logits.masked_fill(context_mask == 0, float('-inf'))
                else:
                    start_logits_for_loss = start_logits
                    end_logits_for_loss = end_logits

                loss_fct = nn.CrossEntropyLoss()
                start_positions = start_positions.clamp(0, input_ids.size(1) - 1)
                end_positions = end_positions.clamp(0, input_ids.size(1) - 1)
                start_loss = loss_fct(start_logits_for_loss, start_positions)
                end_loss = loss_fct(end_logits_for_loss, end_positions)
                total_loss = (start_loss + end_loss) / 2

            return total_loss, start_logits, end_logits

    class ToyLLMForPretraining(nn.Module):
        """Wrapper for the pre-training phase, includes the Language Model head."""

        def __init__(self, base_model: ToyLLM, tie_weights: bool = True):
            super().__init__()
            self.llm = base_model
            self.lm_head = nn.Linear(self.llm.config.hidden_size, self.llm.vocab_size, bias=False)

            # Initialize LM head with same distribution as other weights
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

            # Tie weights between token embeddings and LM head (GPT-2 style)
            if tie_weights:
                self.lm_head.weight = self.llm.token_embedding.weight

        def forward(self, input_ids, attention_mask=None, is_causal=False):
            sequence_output = self.llm(
                input_ids, attention_mask=attention_mask, is_causal=is_causal
            )
            logits = self.lm_head(sequence_output)
            return logits

    # ============================================================
    # Dataset Classes (adapted for Modal)
    # ============================================================

    class NQOpenDataset(IterableDataset):
        """Dataset class for nq_open."""

        _cached_dataset = None

        def __init__(self, split: str, cache_dir: str = HF_CACHE_DIR):
            self.split = split
            self.cache_dir = cache_dir

            if NQOpenDataset._cached_dataset is None:
                print(f"Loading nq_open dataset...")
                NQOpenDataset._cached_dataset = load_dataset(
                    "nq_open",
                    cache_dir=self.cache_dir,
                                    )
                print("Successfully loaded nq_open dataset")

            self.dataset = NQOpenDataset._cached_dataset[self.split]
            print(f"Using nq_open {split} split with {len(self.dataset)} examples")

        def __iter__(self):
            for item in self.dataset:
                question = item.get("question", "").strip()
                answer = item.get("answer", [])
                if question and answer:
                    yield f"Question: {question} Answer: {answer[0]}"

    class WikiText103Dataset(IterableDataset):
        """Dataset class for WikiText-103."""

        _cached_dataset = None

        def __init__(self, split: str, cache_dir: str = HF_CACHE_DIR):
            self.split = split
            self.cache_dir = cache_dir
            hf_split = "train" if split == "train" else "validation"

            if WikiText103Dataset._cached_dataset is None:
                print(f"Loading WikiText-103 dataset...")
                WikiText103Dataset._cached_dataset = load_dataset(
                    "wikitext",
                    "wikitext-103-raw-v1",
                    cache_dir=self.cache_dir,
                                    )
                print("Successfully loaded WikiText-103 dataset")

            self.dataset = WikiText103Dataset._cached_dataset[hf_split]
            print(f"Using WikiText-103 {hf_split} split with {len(self.dataset)} examples")

        def __iter__(self):
            for item in self.dataset:
                text = item.get("text", "").strip()
                if text and len(text) > 50 and not text.startswith(" = "):
                    yield text

    class CombinedPretrainDataset(IterableDataset):
        """Combines multiple pre-training datasets by interleaving them."""

        def __init__(self, datasets: List[IterableDataset], weights: List[float] = None, max_samples: int = None):
            self.datasets = datasets
            self.weights = weights if weights else [1.0 / len(datasets)] * len(datasets)
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
            self.max_samples = max_samples  # Cap samples per epoch for scaled training

        def __iter__(self):
            iterators = [iter(ds) for ds in self.datasets]
            exhausted = [False] * len(self.datasets)
            sample_count = 0

            while not all(exhausted):
                # Check if we've hit the max samples limit
                if self.max_samples is not None and sample_count >= self.max_samples:
                    break

                available_indices = [i for i, e in enumerate(exhausted) if not e]
                if not available_indices:
                    break

                available_weights = [self.weights[i] for i in available_indices]
                total_weight = sum(available_weights)
                normalized_weights = [w / total_weight for w in available_weights]

                idx = random.choices(available_indices, weights=normalized_weights, k=1)[0]

                try:
                    yield next(iterators[idx])
                    sample_count += 1
                except StopIteration:
                    exhausted[idx] = True
                    iterators[idx] = iter(self.datasets[idx])

    class SQuADDataset(Dataset):
        """Dataset class for SQuAD."""

        def __init__(self, split: str, tokenizer, max_seq_len: int, cache_dir: str = HF_CACHE_DIR):
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len

            hf_split = "train" if split == "train" else "validation"
            print(f"Loading SQuAD dataset ({hf_split})...")
            self.dataset = load_dataset(
                "squad",
                split=hf_split,
                cache_dir=cache_dir,
                            )
            print(f"Successfully loaded SQuAD {hf_split} with {len(self.dataset)} examples")

            self.processed_data = self._preprocess()

        def _preprocess(self):
            processed = []
            for example in self.dataset:
                context = example["context"]
                question = example["question"]
                answers = example["answers"]

                if not answers["text"]:
                    continue

                answer_text = answers["text"][0]
                char_start = answers["answer_start"][0]
                char_end = char_start + len(answer_text)

                encoding = self.tokenizer(
                    question,
                    context,
                    truncation="only_second",
                    max_length=self.max_seq_len,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

                for i in range(len(encoding["input_ids"])):
                    sequence_ids = encoding.sequence_ids(i)
                    context_indices = [
                        idx for idx, sid in enumerate(sequence_ids) if sid == 1
                    ]
                    if not context_indices:
                        continue

                    offset_mapping = encoding["offset_mapping"][i]
                    token_start_index = -1
                    token_end_index = -1

                    for idx in context_indices:
                        start, end = offset_mapping[idx]
                        if start <= char_start < end:
                            token_start_index = idx
                        if start < char_end <= end:
                            token_end_index = idx

                    if token_start_index != -1 and token_end_index != -1:
                        # Create context mask: 1 for context positions, 0 for question/special tokens
                        context_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
                        for idx in context_indices:
                            context_mask[idx] = 1

                        processed.append(
                            {
                                "input_ids": torch.tensor(
                                    encoding["input_ids"][i], dtype=torch.long
                                ),
                                "attention_mask": torch.tensor(
                                    encoding["attention_mask"][i], dtype=torch.long
                                ),
                                "start_position": torch.tensor(
                                    token_start_index, dtype=torch.long
                                ),
                                "end_position": torch.tensor(
                                    token_end_index, dtype=torch.long
                                ),
                                "context_mask": context_mask,
                            }
                        )
                        break

            print(f"Finished processing SQuAD. Found {len(processed)} valid QA examples.")
            return processed

        def __len__(self):
            return len(self.processed_data)

        def __getitem__(self, idx):
            return self.processed_data[idx]

    class QASRLDataset(Dataset):
        """Dataset class for QA-SRL loaded from HuggingFace."""

        def __init__(self, split: str, tokenizer, max_seq_len: int, cache_dir: str = HF_CACHE_DIR):
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len

            hf_split = "train" if split == "train" else "validation"
            print(f"Loading QA-SRL dataset ({hf_split}) from HuggingFace...")
            self.dataset = load_dataset(
                "qa_srl",
                split=hf_split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            print(f"Successfully loaded QA-SRL {hf_split} with {len(self.dataset)} examples")

            self.processed_data = self._preprocess()

        def _find_answer_span(self, context: str, answer_text: str):
            """
            Find the character span of the answer in the context.

            Strategy:
            1. Try exact case match first
            2. If no exact match, try case-insensitive match
            3. If multiple matches exist, return None (skip ambiguous examples)

            Returns:
                (char_start, char_end) tuple or None if not found/ambiguous
            """
            # Try exact match first
            exact_start = context.find(answer_text)
            if exact_start != -1:
                # Check for multiple exact matches
                second_match = context.find(answer_text, exact_start + 1)
                if second_match == -1:
                    return exact_start, exact_start + len(answer_text)
                # Multiple exact matches - skip this example
                return None

            # Try case-insensitive match
            context_lower = context.lower()
            answer_lower = answer_text.lower()
            ci_start = context_lower.find(answer_lower)
            if ci_start == -1:
                return None

            # Check for multiple case-insensitive matches
            second_ci_match = context_lower.find(answer_lower, ci_start + 1)
            if second_ci_match != -1:
                # Multiple matches - skip ambiguous example
                return None

            return ci_start, ci_start + len(answer_text)

        def _preprocess(self):
            processed = []
            # Detailed filtering statistics for paper reporting
            stats = {
                "total_raw_examples": len(self.dataset),
                "total_answer_instances": 0,
                "skipped_no_answers": 0,
                "skipped_ambiguous_span": 0,  # Multiple matches in context
                "skipped_answer_not_found": 0,  # Answer text not in context
                "skipped_no_context_tokens": 0,  # Tokenization produced no context
                "skipped_span_alignment_failed": 0,  # Could not align char span to tokens
                "valid_examples": 0,
            }

            for example in self.dataset:
                context = example["sentence"]
                # Join question tokens, filtering out placeholder tokens
                question = " ".join(
                    [token for token in example["question"] if token != "_"]
                )
                if not example.get("answers"):
                    stats["skipped_no_answers"] += 1
                    continue

                for answer_text in example["answers"]:
                    stats["total_answer_instances"] += 1
                    span = self._find_answer_span(context, answer_text)
                    if span is None:
                        # Distinguish between "not found" and "ambiguous"
                        context_lower = context.lower()
                        answer_lower = answer_text.lower()
                        if context_lower.find(answer_lower) == -1:
                            stats["skipped_answer_not_found"] += 1
                        else:
                            stats["skipped_ambiguous_span"] += 1
                        continue
                    char_start, char_end = span

                    encoding = self.tokenizer(
                        question,
                        context,
                        truncation="only_second",
                        max_length=self.max_seq_len,
                        stride=128,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length",
                    )

                    example_added = False
                    for i in range(len(encoding["input_ids"])):
                        sequence_ids = encoding.sequence_ids(i)
                        context_indices = [
                            idx for idx, sid in enumerate(sequence_ids) if sid == 1
                        ]
                        if not context_indices:
                            continue

                        offset_mapping = encoding["offset_mapping"][i]
                        token_start_index = -1
                        token_end_index = -1

                        for idx in context_indices:
                            start, end = offset_mapping[idx]
                            if start <= char_start < end:
                                token_start_index = idx
                            if start < char_end <= end:
                                token_end_index = idx

                        if token_start_index != -1 and token_end_index != -1:
                            # Create context mask: 1 for context positions, 0 for question/special tokens
                            context_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
                            for idx in context_indices:
                                context_mask[idx] = 1

                            processed.append(
                                {
                                    "input_ids": torch.tensor(
                                        encoding["input_ids"][i], dtype=torch.long
                                    ),
                                    "attention_mask": torch.tensor(
                                        encoding["attention_mask"][i], dtype=torch.long
                                    ),
                                    "start_position": torch.tensor(
                                        token_start_index, dtype=torch.long
                                    ),
                                    "end_position": torch.tensor(
                                        token_end_index, dtype=torch.long
                                    ),
                                    "context_mask": context_mask,
                                }
                            )
                            example_added = True
                            break

                    if not example_added:
                        # Check why it failed
                        sequence_ids = encoding.sequence_ids(0) if encoding["input_ids"] else []
                        context_indices = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]
                        if not context_indices:
                            stats["skipped_no_context_tokens"] += 1
                        else:
                            stats["skipped_span_alignment_failed"] += 1

            stats["valid_examples"] = len(processed)
            self.filtering_stats = stats  # Store for later access

            # Print detailed statistics
            print(f"\nQA-SRL Dataset Filtering Statistics:")
            print(f"  Raw examples in dataset: {stats['total_raw_examples']}")
            print(f"  Total answer instances: {stats['total_answer_instances']}")
            print(f"  Skipped (no answers): {stats['skipped_no_answers']}")
            print(f"  Skipped (answer not in context): {stats['skipped_answer_not_found']}")
            print(f"  Skipped (ambiguous - multiple matches): {stats['skipped_ambiguous_span']}")
            print(f"  Skipped (no context tokens after tokenization): {stats['skipped_no_context_tokens']}")
            print(f"  Skipped (span alignment failed): {stats['skipped_span_alignment_failed']}")
            print(f"  Valid examples: {stats['valid_examples']}")
            total_skipped = (stats['skipped_no_answers'] + stats['skipped_answer_not_found'] +
                           stats['skipped_ambiguous_span'] + stats['skipped_no_context_tokens'] +
                           stats['skipped_span_alignment_failed'])
            if stats['total_answer_instances'] > 0:
                skip_rate = 100 * total_skipped / stats['total_answer_instances']
                print(f"  Skip rate: {skip_rate:.1f}%\n")

            return processed

        def __len__(self):
            return len(self.processed_data)

        def __getitem__(self, idx):
            return self.processed_data[idx]

    # ============================================================
    # Helper Functions
    # ============================================================

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def create_clm_inputs_and_labels(token_ids, pad_token_id):
        inputs = token_ids[:, :-1].contiguous()
        labels = token_ids[:, 1:].clone().contiguous()
        labels[labels == pad_token_id] = -100
        return inputs, labels

    def collate_batch_clm(batch, tokenizer, max_seq_len, device):
        tokenized_texts = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        if tokenized_texts["input_ids"].nelement() == 0:
            return None, None, None

        clm_inputs, clm_labels = create_clm_inputs_and_labels(
            tokenized_texts["input_ids"], tokenizer.pad_token_id
        )

        attention_mask = tokenized_texts["attention_mask"][:, :-1].contiguous()

        return (
            clm_inputs.to(device),
            clm_labels.to(device),
            attention_mask.to(device),
        )

    def collate_batch_qa(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        start_positions = torch.stack([item["start_position"] for item in batch])
        end_positions = torch.stack([item["end_position"] for item in batch])
        context_mask = torch.stack([item["context_mask"] for item in batch])
        return input_ids, attention_mask, start_positions, end_positions, context_mask

    def validate_pretrain_epoch(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for clm_inputs, clm_labels, attention_masks in dataloader:
                if clm_inputs is None:
                    continue
                output_logits = model(
                    clm_inputs, attention_mask=attention_masks, is_causal=True
                )
                loss = criterion(
                    output_logits.view(-1, model.llm.vocab_size), clm_labels.view(-1)
                )
                total_loss += loss.item()
                num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate_qa_metrics(model, dataloader, device):
        """Evaluates the model on a QA dataset using Exact Match and F1 score.

        Uses context_mask to ensure predictions are only made within the context
        portion of the input, not in the question or special token positions.
        """
        model.eval()
        all_pred_start, all_pred_end = [], []
        all_true_start, all_true_end = [], []

        with torch.no_grad():
            for input_ids, attention_mask, start_pos, end_pos, context_mask in dataloader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                start_pos, end_pos = start_pos.to(device), end_pos.to(device)
                context_mask = context_mask.to(device)

                _, start_logits, end_logits = model(
                    input_ids, attention_mask=attention_mask
                )

                # Mask out non-context positions before taking argmax
                # Set logits to -inf for question/special token positions
                masked_start_logits = start_logits.masked_fill(context_mask == 0, float('-inf'))
                masked_end_logits = end_logits.masked_fill(context_mask == 0, float('-inf'))

                all_pred_start.append(torch.argmax(masked_start_logits, dim=1).cpu())
                all_pred_end.append(torch.argmax(masked_end_logits, dim=1).cpu())
                all_true_start.append(start_pos.cpu())
                all_true_end.append(end_pos.cpu())

        all_pred_start = torch.cat(all_pred_start)
        all_pred_end = torch.cat(all_pred_end)
        all_true_start = torch.cat(all_true_start)
        all_true_end = torch.cat(all_true_end)

        exact_match = (
            ((all_pred_start == all_true_start) & (all_pred_end == all_true_end))
            .float()
            .mean()
            .item()
        )

        f1_scores = []
        for i in range(len(all_true_start)):
            true_start, true_end = all_true_start[i].item(), all_true_end[i].item()
            pred_start, pred_end = all_pred_start[i].item(), all_pred_end[i].item()

            if pred_start > pred_end:
                pred_tokens = set()
            else:
                pred_tokens = set(range(pred_start, pred_end + 1))

            true_tokens = set(range(true_start, true_end + 1))
            common_tokens = len(pred_tokens.intersection(true_tokens))

            if common_tokens == 0:
                f1_scores.append(0.0)
                continue

            precision = common_tokens / len(pred_tokens)
            recall = common_tokens / len(true_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)

        final_f1 = sum(f1_scores) / len(f1_scores)
        return exact_match, final_f1

    def finetune_qa_epoch(model, dataloader, optimizer, device, epoch_num, gradient_accumulation_steps=1, log_interval=50, scheduler=None):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (input_ids, attention_mask, start_pos, end_pos, context_mask) in enumerate(dataloader):
            input_ids, attention_mask, start_pos, end_pos, context_mask = (
                input_ids.to(device),
                attention_mask.to(device),
                start_pos.to(device),
                end_pos.to(device),
                context_mask.to(device),
            )
            loss, _, _ = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
                context_mask=context_mask,
            )
            if loss is None:
                continue

            loss = loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()  # Step warmup scheduler per optimizer step
                optimizer.zero_grad()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Finetune Epoch {epoch_num} | Batch {batch_idx}/{len(dataloader)} | Avg Loss: {avg_loss:.4f}"
                )
        return total_loss / len(dataloader)

    def validate_qa_epoch(model, dataloader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, attention_mask, start_pos, end_pos, context_mask in dataloader:
                input_ids, attention_mask, start_pos, end_pos, context_mask = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    start_pos.to(device),
                    end_pos.to(device),
                    context_mask.to(device),
                )
                loss, _, _ = model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                    context_mask=context_mask,
                )
                if loss is not None:
                    total_loss += loss.item()

        val_acc, val_f1 = evaluate_qa_metrics(model, dataloader, device)
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        return avg_loss, val_acc, val_f1

    def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, best_val_loss, epochs_no_improve, stage="pretrain", best_val_f1=None, model_dir=None):
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'stage': stage,
            'model_dir': model_dir,
        }
        if best_val_f1 is not None:
            checkpoint['best_val_f1'] = best_val_f1
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(path, model, optimizer, scheduler=None, device='cuda'):
        if not os.path.exists(path):
            return None

        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Resumed from epoch {checkpoint['epoch']}, global_step {checkpoint['global_step']}")
        return checkpoint

    # ============================================================
    # Main Training Logic
    # ============================================================

    # Load configuration
    config = CONFIGS[config_name]
    llm_params = config['llm_config']
    train_params = config['training_params']
    print(f"--- Loaded configuration '{config_name}' ---")

    # Setup device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load tokenizer
    print("Loading tokenizer ('bert-base-uncased')...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=HF_CACHE_DIR)
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_TOKEN_ID = tokenizer.pad_token_id

    # Create model
    model_config = LLMConfig(**llm_params)
    base_llm = ToyLLM(
        config=model_config,
        vocab_size=VOCAB_SIZE,
        max_seq_len=train_params['max_seq_len'],
        pad_token_id=PAD_TOKEN_ID,
        dropout_rate=train_params['dropout_rate'],
    )

    num_params = sum(p.numel() for p in base_llm.parameters() if p.requires_grad)
    print(f"Instantiated Base Model: {model_config.name} with {num_params:,} trainable parameters.")

    # Setup model directory (includes samples_per_epoch for reproducibility)
    samples_per_epoch = train_params.get('samples_per_epoch', None)
    samples_suffix = f"_spe{samples_per_epoch // 1000}k" if samples_per_epoch else "_speFull"

    # Auto-resume: find latest checkpoint for this config if no resume_path provided
    def find_latest_checkpoint(model_name_prefix):
        """Find the most recent checkpoint for this model configuration."""
        models_dir = "/models"
        if not os.path.exists(models_dir):
            return None

        matching_dirs = []
        for dirname in os.listdir(models_dir):
            if dirname.startswith(model_name_prefix):
                checkpoint_path = os.path.join(models_dir, dirname, "checkpoint.pth")
                if os.path.exists(checkpoint_path):
                    mtime = os.path.getmtime(checkpoint_path)
                    matching_dirs.append((checkpoint_path, mtime, dirname))

        if not matching_dirs:
            return None

        # Return the most recently modified checkpoint
        matching_dirs.sort(key=lambda x: x[1], reverse=True)
        return matching_dirs[0][0], matching_dirs[0][2]  # (checkpoint_path, dirname)

    # Add suffixes to distinguish experimental conditions
    frozen_suffix = "_frozen" if freeze_llm else ""
    random_suffix = "_random" if random_baseline else ""

    if resume_path:
        model_dir = os.path.dirname(os.path.join("/models", resume_path))
    else:
        # Check for existing checkpoint to auto-resume from
        model_prefix = f"{model_config.name}{samples_suffix}{frozen_suffix}{random_suffix}"
        latest = find_latest_checkpoint(model_prefix)

        if latest and not random_baseline:  # Don't auto-resume for random baseline runs
            checkpoint_path, dirname = latest
            print(f"\n{'='*60}")
            print(f"AUTO-RESUME: Found existing checkpoint!")
            print(f"  Directory: {dirname}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"{'='*60}\n")
            resume_path = checkpoint_path.replace("/models/", "")
            model_dir = os.path.join("/models", dirname)
        else:
            date_now = time.strftime("%Y%m%d-%H%M%S")
            model_dir = f"/models/{model_config.name}{samples_suffix}{frozen_suffix}{random_suffix}_{date_now}"

    PRETRAINED_MODEL_PATH = os.path.join(model_dir, "toy_llm_unified_pretrained.pth")
    FINETUNED_MODEL_PATH = os.path.join(model_dir, "toy_llm_qasrl_finetuned.pth")
    CHECKPOINT_PATH = os.path.join(model_dir, "checkpoint.pth")

    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")

    # ============================================================
    # Pre-training Phase
    # ============================================================

    if not skip_pretrain:
        # Get samples_per_epoch for scaled training
        samples_per_epoch = train_params.get('samples_per_epoch', None)
        if samples_per_epoch:
            print(f"Using scaled training: {samples_per_epoch:,} samples per epoch")

        # Select pretraining data based on flag
        if pretrain_data == "wiki+nq":
            print(f"\n--- Starting CLM Pre-training on WikiText-103 + nq_open ---")
            print("WARNING: nq_open contains QA-format text which may leak task patterns into pretraining")
            nq_train = NQOpenDataset(split="train")
            wiki_train = WikiText103Dataset(split="train")
            train_dataset_clm = CombinedPretrainDataset(
                [nq_train, wiki_train],
                weights=[0.3, 0.7],
                max_samples=samples_per_epoch
            )

            nq_val = NQOpenDataset(split="validation")
            wiki_val = WikiText103Dataset(split="validation")
            val_dataset_clm = CombinedPretrainDataset(
                [nq_val, wiki_val],
                weights=[0.3, 0.7]
                # No max_samples for validation - use full validation set
            )
        else:  # Default: wiki only (cleaner for emergence research)
            print(f"\n--- Starting CLM Pre-training on WikiText-103 only ---")
            print("Using WikiText-103 only (no QA-format data) for clean emergence testing")
            wiki_train = WikiText103Dataset(split="train")
            wiki_val = WikiText103Dataset(split="validation")

            # Wrap in a simple dataset with max_samples support
            class LimitedIterableDataset(IterableDataset):
                def __init__(self, dataset, max_samples=None):
                    self.dataset = dataset
                    self.max_samples = max_samples

                def __iter__(self):
                    count = 0
                    for item in self.dataset:
                        if self.max_samples and count >= self.max_samples:
                            break
                        yield item
                        count += 1

            train_dataset_clm = LimitedIterableDataset(wiki_train, max_samples=samples_per_epoch)
            val_dataset_clm = wiki_val

        pretrain_model = ToyLLMForPretraining(base_llm).to(DEVICE)

        collate_fn_clm = lambda batch: collate_batch_clm(
            batch, tokenizer, train_params['max_seq_len'], DEVICE
        )
        train_dataloader_clm = DataLoader(
            train_dataset_clm,
            batch_size=train_params['batch_size_pretrain'],
            collate_fn=collate_fn_clm,
        )
        val_dataloader_clm = DataLoader(
            val_dataset_clm,
            batch_size=train_params['batch_size_pretrain'],
            collate_fn=collate_fn_clm,
        )

        optimizer_pretrain = optim.AdamW(
            pretrain_model.parameters(), lr=train_params['pretrain_lr'], weight_decay=0.01
        )
        scheduler_pretrain = ReduceLROnPlateau(
            optimizer_pretrain, mode="min", factor=0.5, patience=2
        )
        criterion_clm = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        global_step = 0
        start_epoch = 1

        pretrain_grad_accum = train_params.get('gradient_accumulation_steps', 1)

        # Resume from checkpoint if provided
        if resume_path:
            full_resume_path = os.path.join("/models", resume_path)
            if os.path.exists(full_resume_path):
                checkpoint = load_checkpoint(full_resume_path, pretrain_model, optimizer_pretrain, scheduler_pretrain, DEVICE)
                if checkpoint and checkpoint.get('stage') == 'pretrain':
                    start_epoch = checkpoint['epoch'] + 1
                    global_step = checkpoint['global_step']
                    best_val_loss = checkpoint['best_val_loss']
                    epochs_no_improve = checkpoint['epochs_no_improve']
                elif checkpoint and checkpoint.get('stage') == 'finetune':
                    print("Checkpoint is from fine-tuning stage. Skipping pre-training.")
                    skip_pretrain = True

        if not skip_pretrain:
            timeout_triggered = False
            for epoch in range(start_epoch, train_params['num_pretrain_epochs'] + 1):
                pretrain_model.train()
                total_train_loss = 0
                optimizer_pretrain.zero_grad()

                for batch_idx, (clm_inputs, clm_labels, attention_masks) in enumerate(train_dataloader_clm):
                    global_step += 1
                    if clm_inputs is None:
                        continue

                    # Check for timeout periodically
                    if batch_idx % TIMEOUT_CHECK_INTERVAL == 0 and is_approaching_timeout():
                        print(f"\n{'='*60}")
                        print(f"TIMEOUT APPROACHING after {get_elapsed_time_str()}")
                        print(f"Saving checkpoint and spawning continuation...")
                        print(f"{'='*60}")

                        # Save checkpoint
                        save_checkpoint(
                            CHECKPOINT_PATH, pretrain_model, optimizer_pretrain, scheduler_pretrain,
                            epoch, global_step, best_val_loss, epochs_no_improve, stage="pretrain",
                            model_dir=model_dir
                        )
                        models_volume.commit()

                        # Get relative checkpoint path for spawning
                        relative_checkpoint = CHECKPOINT_PATH.replace("/models/", "")

                        # Spawn continuation
                        train_fn.spawn(
                            config_name=config_name,
                            skip_pretrain=False,
                            skip_finetune=skip_finetune,
                            resume_path=relative_checkpoint,
                            checkpoint_interval=checkpoint_interval,
                            _continuation_count=_continuation_count + 1,
                        )

                        print(f"Continuation job spawned. This job will now exit.")
                        timeout_triggered = True
                        return {
                            "status": "continued",
                            "model_dir": model_dir,
                            "checkpoint_path": CHECKPOINT_PATH,
                            "continuation_count": _continuation_count + 1,
                            "pretrained_path": PRETRAINED_MODEL_PATH,
                            "finetuned_path": FINETUNED_MODEL_PATH,
                        }

                    if global_step < train_params['warmup_steps']:
                        lr_scale = global_step / train_params['warmup_steps']
                        for param_group in optimizer_pretrain.param_groups:
                            param_group["lr"] = lr_scale * train_params['pretrain_lr']

                    output_logits = pretrain_model(
                        clm_inputs, attention_mask=attention_masks, is_causal=True
                    )
                    loss = criterion_clm(
                        output_logits.view(-1, pretrain_model.llm.vocab_size),
                        clm_labels.view(-1),
                    )

                    loss = loss / pretrain_grad_accum
                    loss.backward()
                    total_train_loss += loss.item() * pretrain_grad_accum

                    if (batch_idx + 1) % pretrain_grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 1.0)
                        optimizer_pretrain.step()
                        optimizer_pretrain.zero_grad()

                    if batch_idx % 500 == 0 and batch_idx > 0:
                        avg_loss_so_far = total_train_loss / (batch_idx + 1)
                        current_lr = optimizer_pretrain.param_groups[0]["lr"]
                        print(f"  [Epoch {epoch}] Batch {batch_idx} | Loss: {avg_loss_so_far:.4f} | LR: {current_lr:.6f} | Elapsed: {get_elapsed_time_str()}")

                if timeout_triggered:
                    break

                avg_train_loss = total_train_loss / (batch_idx + 1)

                avg_val_loss = validate_pretrain_epoch(
                    pretrain_model, val_dataloader_clm, criterion_clm, DEVICE
                )
                current_lr = optimizer_pretrain.param_groups[0]["lr"]
                val_perplexity = math.exp(avg_val_loss)
                print(
                    f"--- End of Pre-train Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val PPL: {val_perplexity:.2f} | LR: {current_lr:.6f} ---"
                )

                # Save epoch snapshot for temporal analysis (CKA during pretraining)
                pretrain_snapshot_path = os.path.join(model_dir, f"pretrain_epoch_{epoch}.pth")
                torch.save(pretrain_model.state_dict(), pretrain_snapshot_path)
                print(f"Saved pretrain epoch snapshot to {pretrain_snapshot_path}")

                scheduler_pretrain.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(pretrain_model.state_dict(), PRETRAINED_MODEL_PATH)
                    print(f"Validation loss improved. Saving model to {PRETRAINED_MODEL_PATH}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= train_params['pretrain_patience']:
                    print(f"\nEarly stopping triggered after {train_params['pretrain_patience']} epochs without improvement.")
                    save_checkpoint(
                        CHECKPOINT_PATH, pretrain_model, optimizer_pretrain, scheduler_pretrain,
                        epoch, global_step, best_val_loss, epochs_no_improve, stage="pretrain",
                        model_dir=model_dir
                    )
                    break

                # Check timeout at end of epoch too
                if is_approaching_timeout():
                    print(f"\n{'='*60}")
                    print(f"TIMEOUT APPROACHING at end of epoch {epoch} after {get_elapsed_time_str()}")
                    print(f"Saving checkpoint and spawning continuation...")
                    print(f"{'='*60}")

                    save_checkpoint(
                        CHECKPOINT_PATH, pretrain_model, optimizer_pretrain, scheduler_pretrain,
                        epoch, global_step, best_val_loss, epochs_no_improve, stage="pretrain",
                        model_dir=model_dir
                    )
                    models_volume.commit()

                    relative_checkpoint = CHECKPOINT_PATH.replace("/models/", "")
                    train_fn.spawn(
                        config_name=config_name,
                        skip_pretrain=False,
                        skip_finetune=skip_finetune,
                        resume_path=relative_checkpoint,
                        checkpoint_interval=checkpoint_interval,
                        _continuation_count=_continuation_count + 1,
                    )

                    print(f"Continuation job spawned. This job will now exit.")
                    return {
                        "status": "continued",
                        "model_dir": model_dir,
                        "checkpoint_path": CHECKPOINT_PATH,
                        "continuation_count": _continuation_count + 1,
                        "pretrained_path": PRETRAINED_MODEL_PATH,
                        "finetuned_path": FINETUNED_MODEL_PATH,
                    }

                if epoch % checkpoint_interval == 0:
                    save_checkpoint(
                        CHECKPOINT_PATH, pretrain_model, optimizer_pretrain, scheduler_pretrain,
                        epoch, global_step, best_val_loss, epochs_no_improve, stage="pretrain",
                        model_dir=model_dir
                    )

            if not timeout_triggered:
                print(f"\nLoading best pretrained model from {PRETRAINED_MODEL_PATH}")
                pretrain_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

            # Commit volume to persist data
            models_volume.commit()

    # ============================================================
    # Fine-tuning Phase
    # ============================================================

    if not skip_finetune:
        use_squad = train_params.get('use_squad_finetune', False)

        if use_squad:
            print(f"\n--- Starting Fine-tuning on SQuAD + QA-SRL ---")
        else:
            print(f"\n--- Starting Fine-tuning on QA-SRL ---")

        qa_model = ToyLLMForQuestionAnswering(base_llm).to(DEVICE)

        # Random baseline: skip loading pretrained weights to establish floor performance
        if random_baseline:
            print("\n*** RANDOM BASELINE: Using randomly initialized LLM (no pretrained weights) ***")
            print("This establishes a floor for the emergence gap metric.")
        else:
            try:
                pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
                qa_model.llm.load_state_dict(
                    {
                        k.replace("llm.", ""): v
                        for k, v in pretrained_dict.items()
                        if k.startswith("llm.")
                    }
                )
                print("Successfully loaded pre-trained weights into the QA model.")
            except FileNotFoundError:
                print(f"WARNING: Pre-trained model not found. Fine-tuning from random init.")

        # Freeze LLM if requested (linear probe mode)
        if freeze_llm:
            print("\n*** FREEZE MODE: Freezing LLM weights, only training QA head ***")
            for param in qa_model.llm.parameters():
                param.requires_grad = False
            trainable_params = sum(p.numel() for p in qa_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in qa_model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Load QA-SRL dataset (primary fine-tuning dataset)
        qa_train_dataset = QASRLDataset(
            split="train", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
        )
        qa_val_dataset = QASRLDataset(
            split="validation", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
        )

        # Limit training samples for few-shot experiments
        if finetune_samples is not None and finetune_samples < len(qa_train_dataset):
            print(f"\n*** FEW-SHOT MODE: Using only {finetune_samples} training samples ***")
            from torch.utils.data import Subset
            indices = list(range(finetune_samples))
            qa_train_dataset = Subset(qa_train_dataset, indices)

        # Optionally add SQuAD dataset for more training data
        if use_squad:
            squad_train_dataset = SQuADDataset(
                split="train", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
            )
            squad_val_dataset = SQuADDataset(
                split="validation", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
            )
            # Combine datasets
            combined_train_dataset = ConcatDataset([qa_train_dataset, squad_train_dataset])
            combined_val_dataset = ConcatDataset([qa_val_dataset, squad_val_dataset])
            print(f"Combined training data: {len(combined_train_dataset)} examples (QA-SRL: {len(qa_train_dataset)}, SQuAD: {len(squad_train_dataset)})")
            print(f"Combined validation data: {len(combined_val_dataset)} examples")

            qa_train_dataloader = DataLoader(
                combined_train_dataset,
                batch_size=train_params['batch_size_qa'],
                shuffle=True,
                collate_fn=collate_batch_qa,
                num_workers=4,
                pin_memory=True,
            )
            qa_val_dataloader = DataLoader(
                combined_val_dataset,
                batch_size=train_params['batch_size_qa'],
                shuffle=False,
                collate_fn=collate_batch_qa,
                num_workers=4,
                pin_memory=True,
            )
        else:
            qa_train_dataloader = DataLoader(
                qa_train_dataset,
                batch_size=train_params['batch_size_qa'],
                shuffle=True,
                collate_fn=collate_batch_qa,
                num_workers=4,
                pin_memory=True,
            )
            qa_val_dataloader = DataLoader(
                qa_val_dataset,
                batch_size=train_params['batch_size_qa'],
                shuffle=False,
                collate_fn=collate_batch_qa,
                num_workers=4,
                pin_memory=True,
            )

        optimizer_finetune = optim.AdamW(qa_model.parameters(), lr=train_params['finetune_lr'], weight_decay=0.01)

        gradient_accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        finetune_patience = train_params.get('finetune_patience', 5)
        finetune_warmup_steps = train_params.get('finetune_warmup_steps', 0)

        steps_per_epoch = len(qa_train_dataloader) // gradient_accumulation_steps
        total_training_steps = steps_per_epoch * train_params['num_finetune_epochs']

        if finetune_warmup_steps > 0:
            scheduler_finetune = get_linear_schedule_with_warmup(
                optimizer_finetune,
                num_warmup_steps=finetune_warmup_steps,
                num_training_steps=total_training_steps
            )
        else:
            scheduler_finetune = ReduceLROnPlateau(
                optimizer_finetune, mode="min", factor=0.5, patience=2
            )

        print(f"Starting fine-tuning for up to {train_params['num_finetune_epochs']} epochs...")

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        epochs_no_improve = 0
        global_finetune_step = 0
        start_epoch_finetune = 1

        FINETUNE_CHECKPOINT_PATH = os.path.join(model_dir, "finetune_checkpoint.pth")

        for epoch_num in range(start_epoch_finetune, train_params['num_finetune_epochs'] + 1):
            # Pass scheduler only for warmup (LambdaLR), not for ReduceLROnPlateau
            warmup_scheduler = scheduler_finetune if finetune_warmup_steps > 0 else None
            avg_train_loss = finetune_qa_epoch(
                qa_model, qa_train_dataloader, optimizer_finetune, DEVICE, epoch_num,
                gradient_accumulation_steps=gradient_accumulation_steps,
                scheduler=warmup_scheduler
            )
            global_finetune_step += steps_per_epoch

            avg_val_loss, val_acc, val_f1 = validate_qa_epoch(qa_model, qa_val_dataloader, DEVICE)

            if finetune_warmup_steps == 0:
                scheduler_finetune.step(avg_val_loss)

            current_lr = optimizer_finetune.param_groups[0]['lr']
            print(
                f"--- End of Finetune Epoch {epoch_num} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
                f"| LR: {current_lr:.6f} | Elapsed: {get_elapsed_time_str()} ---"
            )

            # Save epoch snapshot for temporal analysis (CKA, layer-wise probing)
            epoch_snapshot_path = os.path.join(model_dir, f"finetune_epoch_{epoch_num}.pth")
            torch.save(qa_model.state_dict(), epoch_snapshot_path)
            print(f"Saved epoch snapshot to {epoch_snapshot_path}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(qa_model.state_dict(), FINETUNED_MODEL_PATH)
                print(f"Validation loss improved. Saving best model to {FINETUNED_MODEL_PATH}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= finetune_patience:
                print(f"\nEarly stopping triggered after {finetune_patience} epochs without improvement.")
                break

            # Check for timeout at end of each finetuning epoch
            if is_approaching_timeout():
                print(f"\n{'='*60}")
                print(f"TIMEOUT APPROACHING during finetuning after {get_elapsed_time_str()}")
                print(f"Saving checkpoint and spawning continuation...")
                print(f"{'='*60}")

                save_checkpoint(
                    FINETUNE_CHECKPOINT_PATH, qa_model, optimizer_finetune,
                    scheduler_finetune if finetune_warmup_steps > 0 else None,
                    epoch_num, global_finetune_step, best_val_loss, epochs_no_improve,
                    stage="finetune", best_val_f1=best_val_f1, model_dir=model_dir
                )
                models_volume.commit()

                relative_checkpoint = FINETUNE_CHECKPOINT_PATH.replace("/models/", "")
                train_fn.spawn(
                    config_name=config_name,
                    skip_pretrain=True,  # Pretraining is done
                    skip_finetune=False,
                    resume_path=relative_checkpoint,
                    checkpoint_interval=checkpoint_interval,
                    _continuation_count=_continuation_count + 1,
                )

                print(f"Continuation job spawned. This job will now exit.")
                return {
                    "status": "continued",
                    "model_dir": model_dir,
                    "checkpoint_path": FINETUNE_CHECKPOINT_PATH,
                    "continuation_count": _continuation_count + 1,
                    "pretrained_path": PRETRAINED_MODEL_PATH,
                    "finetuned_path": FINETUNED_MODEL_PATH,
                }

            if epoch_num % checkpoint_interval == 0:
                save_checkpoint(
                    FINETUNE_CHECKPOINT_PATH, qa_model, optimizer_finetune,
                    scheduler_finetune if finetune_warmup_steps > 0 else None,
                    epoch_num, global_finetune_step, best_val_loss, epochs_no_improve,
                    stage="finetune", best_val_f1=best_val_f1, model_dir=model_dir
                )

        print("\nFine-tuning finished.")
        print(f"Best fine-tuned model saved to {FINETUNED_MODEL_PATH}")

        # Final evaluation
        qa_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
        final_val_acc, final_val_f1 = evaluate_qa_metrics(qa_model, qa_val_dataloader, DEVICE)
        print(f"Final Validation Accuracy: {final_val_acc:.4f} | Final F1: {final_val_f1:.4f}")

        # Save stats including QA-SRL filtering statistics for paper reporting
        stats = {
            "config_name": config_name,
            "finetune_epochs": epoch_num,
            "best_validation_loss": best_val_loss,
            "final_validation_accuracy": final_val_acc,
            "final_validation_f1": final_val_f1,
            "freeze_llm": freeze_llm,
            "random_baseline": random_baseline,
            "finetune_samples_limit": finetune_samples,
        }

        # Include QA-SRL filtering statistics if available
        # Handle both direct QASRLDataset and Subset wrapper cases
        train_dataset_for_stats = qa_train_dataset
        if hasattr(qa_train_dataset, 'dataset'):  # It's a Subset
            train_dataset_for_stats = qa_train_dataset.dataset
        if hasattr(train_dataset_for_stats, 'filtering_stats'):
            stats["qasrl_train_filtering"] = train_dataset_for_stats.filtering_stats

        val_dataset_for_stats = qa_val_dataset
        if hasattr(qa_val_dataset, 'dataset'):  # It's a Subset
            val_dataset_for_stats = qa_val_dataset.dataset
        if hasattr(val_dataset_for_stats, 'filtering_stats'):
            stats["qasrl_val_filtering"] = val_dataset_for_stats.filtering_stats

        stats_path = os.path.join(model_dir, "finetune_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Final training statistics saved to '{stats_path}'")

        # Commit volume to persist models
        models_volume.commit()

    return {
        "model_dir": model_dir,
        "pretrained_path": PRETRAINED_MODEL_PATH,
        "finetuned_path": FINETUNED_MODEL_PATH,
    }


# GPU-specific wrapper functions - these call the shared implementation
@app.function(
    image=image,
    gpu="A10G",  # For tiny, small, base configs
    timeout=86400,
    volumes={"/data": data_volume, "/models": models_volume},
)
def train_a10g(
    config_name: str = "tiny",
    skip_pretrain: bool = False,
    skip_finetune: bool = False,
    resume_path: str = None,
    checkpoint_interval: int = 1,
    freeze_llm: bool = False,
    finetune_samples: int = None,
    pretrain_data: str = "wiki",
    random_baseline: bool = False,
    _continuation_count: int = 0,
):
    """Training function using A10G GPU (for tiny, small, base configs)."""
    return _train_impl(
        config_name=config_name,
        skip_pretrain=skip_pretrain,
        skip_finetune=skip_finetune,
        resume_path=resume_path,
        checkpoint_interval=checkpoint_interval,
        freeze_llm=freeze_llm,
        finetune_samples=finetune_samples,
        pretrain_data=pretrain_data,
        random_baseline=random_baseline,
        _continuation_count=_continuation_count,
        train_fn=train_a10g,  # Pass self for spawning continuations
    )


@app.function(
    image=image,
    gpu="A100-80GB",  # For medium config - 80GB needed for batch=96
    timeout=86400,
    volumes={"/data": data_volume, "/models": models_volume},
)
def train_a100(
    config_name: str = "medium",
    skip_pretrain: bool = False,
    skip_finetune: bool = False,
    resume_path: str = None,
    checkpoint_interval: int = 1,
    freeze_llm: bool = False,
    finetune_samples: int = None,
    pretrain_data: str = "wiki",
    random_baseline: bool = False,
    _continuation_count: int = 0,
):
    """Training function using A100 GPU (for medium config)."""
    return _train_impl(
        config_name=config_name,
        skip_pretrain=skip_pretrain,
        skip_finetune=skip_finetune,
        resume_path=resume_path,
        checkpoint_interval=checkpoint_interval,
        freeze_llm=freeze_llm,
        finetune_samples=finetune_samples,
        pretrain_data=pretrain_data,
        random_baseline=random_baseline,
        _continuation_count=_continuation_count,
        train_fn=train_a100,  # Pass self for spawning continuations
    )


# Alias for backwards compatibility - routes to appropriate GPU automatically
def train_remote(config_name, **kwargs):
    """Route to appropriate GPU based on config."""
    if config_name == "medium":
        return train_a100.remote(config_name=config_name, **kwargs)
    else:
        return train_a10g.remote(config_name=config_name, **kwargs)


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def list_models():
    """List all saved models in the volume."""
    import os

    models_dir = "/models"
    if not os.path.exists(models_dir):
        return []

    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            models.append({
                "name": item,
                "files": files,
            })
    return models


@app.function(
    image=image,
    volumes={"/models": models_volume},
)
def download_model(model_name: str, filename: str):
    """Download a specific model file."""
    import os

    file_path = os.path.join("/models", model_name, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    with open(file_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    config_name: str = "tiny",
    skip_pretrain: bool = False,
    skip_finetune: bool = False,
    resume: str = None,
    checkpoint_interval: int = 1,
    gpu: str = "A10G",
    list_saved: bool = False,
    freeze_llm: bool = False,
    finetune_samples: int = None,
    pretrain_data: str = "wiki",
    random_baseline: bool = False,
):
    """
    Main entry point for running training on Modal.

    Args:
        config_name: Model configuration ('tiny', 'small', 'base', 'medium')
        skip_pretrain: Skip the pre-training phase
        skip_finetune: Skip the fine-tuning phase
        resume: Path to checkpoint to resume from (relative to /models volume)
        checkpoint_interval: Save checkpoint every N epochs
        gpu: GPU type to use ('T4', 'A10G', 'A100', 'H100')
        list_saved: List saved models instead of training
        freeze_llm: Freeze LLM weights during finetuning (linear probe mode)
        finetune_samples: Limit finetuning to N samples (for few-shot experiments)
        pretrain_data: Pretraining data ('wiki' or 'wiki+nq'). Default 'wiki' for clean emergence testing.
        random_baseline: Use randomly initialized LLM (no pretraining) to establish floor for emergence gap.
    """
    if list_saved:
        models = list_models.remote()
        print("\nSaved models:")
        for model in models:
            print(f"  {model['name']}/")
            for f in model['files']:
                print(f"    - {f}")
        return

    # Auto-select GPU based on config
    selected_gpu = "A100" if config_name == "medium" else "A10G"

    print(f"\n{'='*60}")
    print(f"Starting Modal Training Job")
    print(f"{'='*60}")
    print(f"Config: {config_name}")
    print(f"GPU: {selected_gpu} (auto-selected)")
    print(f"Pretrain data: {pretrain_data}")
    print(f"Skip pretrain: {skip_pretrain}")
    print(f"Skip finetune: {skip_finetune}")
    print(f"Freeze LLM: {freeze_llm}")
    print(f"Random baseline: {random_baseline}")
    if finetune_samples:
        print(f"Finetune samples: {finetune_samples}")
    if resume:
        print(f"Resume from: {resume}")
    print(f"{'='*60}\n")

    # Route to appropriate GPU based on config
    result = train_remote(
        config_name=config_name,
        skip_pretrain=skip_pretrain,
        skip_finetune=skip_finetune,
        resume_path=resume,
        checkpoint_interval=checkpoint_interval,
        freeze_llm=freeze_llm,
        finetune_samples=finetune_samples,
        pretrain_data=pretrain_data,
        random_baseline=random_baseline,
    )

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Model directory: {result['model_dir']}")
    print(f"Pretrained model: {result['pretrained_path']}")
    print(f"Finetuned model: {result['finetuned_path']}")
    print(f"{'='*60}")