import json
import os
import time
from typing import Iterator, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer
import argparse

DATASET_PATH = "/home/zceccgr/Scratch/downloaded_datasets/qa_srl"


try:
    from llm_scale_up.utils.eval_metrics import evaluate_qa_metrics
except ImportError:
    print("Warning: `evaluate_qa_metrics` not found. Using a mock function.")

    def evaluate_qa_metrics(model, dataloader, device):
        # Mock function returns dummy values if the original is not available
        print("Mock evaluation: Returning 0.0 for accuracy and F1.")
        return 0.0, 0.0


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

    def __repr__(self):
        return (
            f"LLMConfig(name='{self.name}', n_layers={self.n_layers}, "
            f"hidden_size={self.hidden_size}, n_heads={self.n_heads})"
        )


class ToyMultiHeadAttention(nn.Module):
    """Multi-Head Attention module for the ToyLLM."""

    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

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
        self.attention = ToyMultiHeadAttention(hidden_size, n_heads)
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

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        # QA fine-tuning is bidirectional over the context, so is_causal=False
        sequence_output = self.llm(
            input_ids, attention_mask=attention_mask, is_causal=False
        )
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_positions = start_positions.clamp(0, input_ids.size(1) - 1)
            end_positions = end_positions.clamp(0, input_ids.size(1) - 1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits


class ToyLLMForPretraining(nn.Module):
    """Wrapper for the pre-training phase, includes the Language Model head."""

    def __init__(self, base_model: ToyLLM):
        super().__init__()
        self.llm = base_model
        self.lm_head = nn.Linear(self.llm.config.hidden_size, self.llm.vocab_size)

    def forward(self, input_ids, attention_mask=None, is_causal=False):
        sequence_output = self.llm(
            input_ids, attention_mask=attention_mask, is_causal=is_causal
        )
        logits = self.lm_head(sequence_output)
        return logits


def create_clm_inputs_and_labels(token_ids, pad_token_id):
    """Creates inputs and labels for Causal Language Modeling."""
    inputs = token_ids[:, :-1].contiguous()
    labels = token_ids[:, 1:].clone().contiguous()
    labels[labels == pad_token_id] = -100
    return inputs, labels


def collate_batch_clm(batch, tokenizer, max_seq_len, device):
    """Collates a batch of text for Causal Language Modeling."""
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


def validate_pretrain_epoch(model, dataloader, criterion, device):
    """Runs a validation loop for one epoch during pre-training."""
    model.eval()
    total_loss = 0
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
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


class NQOpenDataset(IterableDataset):
    """
    Dataset class for nq_open. Implemented as an IterableDataset to stream data
    and avoid loading the entire dataset into memory.
    """

    DATASET_NAME = "nq_open"

    def __init__(self, split: str, cache_dir: str = None):
        self.split = split
        self.cache_dir = cache_dir
        self.dataset = load_dataset(self.DATASET_NAME, split=self.split, cache_dir=self.cache_dir)

    def __iter__(self):
        for item in self.dataset:
            question = item.get("question", "").strip()
            answer = item.get("answer", [])
            if question and answer:
                yield f"Question: {question} Answer: {answer[0]}"


class QASRLDataset(Dataset):
    """Dataset class for fine-tuning on qa_srl."""

    def __init__(self, split, tokenizer, max_seq_len):
        print(f"Loading and processing qa_srl dataset for '{split}' split...")
        self.dataset = load_from_disk(DATASET_PATH)[split]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.processed_data = self._preprocess()

    def _preprocess(self):
        processed = []
        for example in self.dataset:
            context = example["sentence"]
            question = " ".join(
                [token for token in example["question"] if token != "_"]
            )
            if not example.get("answers"):
                continue

            for answer_text in example["answers"]:
                char_start = context.lower().find(answer_text.lower())
                if char_start == -1:
                    continue
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

                    for idx, (start, end) in zip(context_indices, offset_mapping):
                        if start <= char_start < end:
                            token_start_index = idx
                        if start < char_end <= end:
                            token_end_index = idx

                    if token_start_index != -1 and token_end_index != -1:
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
                            }
                        )
                        break
        print(
            f"Finished processing '{self.dataset.split}'. Found {len(processed)} valid QA examples."
        )
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_batch_qa(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    start_positions = torch.stack([item["start_position"] for item in batch])
    end_positions = torch.stack([item["end_position"] for item in batch])
    return input_ids, attention_mask, start_positions, end_positions


def finetune_qa_epoch(model, dataloader, optimizer, device, epoch_num, gradient_accumulation_steps=1, log_interval=50):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, (input_ids, attention_mask, start_pos, end_pos) in enumerate(
        dataloader
    ):
        input_ids, attention_mask, start_pos, end_pos = (
            input_ids.to(device),
            attention_mask.to(device),
            start_pos.to(device),
            end_pos.to(device),
        )
        loss, _, _ = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_pos,
            end_positions=end_pos,
        )
        if loss is None:
            continue

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps

        # Only update weights after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Finetune Epoch {epoch_num} | Batch {batch_idx}/{len(dataloader)} | Avg Loss: {avg_loss:.4f}"
            )
    return total_loss / len(dataloader)


def validate_qa_epoch(model, dataloader, device):
    """Runs a validation loop for one epoch during fine-tuning."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, start_pos, end_pos in dataloader:
            input_ids, attention_mask, start_pos, end_pos = (
                input_ids.to(device),
                attention_mask.to(device),
                start_pos.to(device),
                end_pos.to(device),
            )
            loss, _, _ = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            if loss is not None:
                total_loss += loss.item()

    val_acc, val_f1 = evaluate_qa_metrics(model, dataloader, device)
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss, val_acc, val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train and fine-tune a ToyLLM.")
    parser.add_argument(
        '--config_path',
        type=str,
        default='/home/zceccgr/Scratch/freezeLLM/llm_scale_up/config.json',
        help='Path to the configuration JSON file.'
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default='tiny',
        choices=['tiny', 'small', 'base', 'medium'],
        help='The name of the configuration to use from the JSON file.'
    )
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        all_configs = json.load(f)

    config = all_configs[args.config_name]
    llm_params = config['llm_config']
    train_params = config['training_params']
    print(f"--- Loaded configuration '{args.config_name}' from '{args.config_path}' ---")

    SKIP_PRETRAIN = False
    SKIP_FINETUNE = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading standard tokenizer ('bert-base-uncased')...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_TOKEN_ID = tokenizer.pad_token_id
    print(f"Standard vocabulary size: {VOCAB_SIZE}")
    print(f"PAD token ID: {PAD_TOKEN_ID}")

    model_config = LLMConfig(**llm_params)
    base_llm = ToyLLM(
        config=model_config,
        vocab_size=VOCAB_SIZE,
        max_seq_len=train_params['max_seq_len'],
        pad_token_id=PAD_TOKEN_ID,
        dropout_rate=train_params['dropout_rate'],
    )

    num_params = sum(p.numel() for p in base_llm.parameters() if p.requires_grad)
    print(
        f"Instantiated Base Model: {model_config.name} with {num_params:,} trainable parameters."
    )

    date_now = time.strftime("%Y%m%d-%H%M%S")
    model_dir = f"models/{model_config.name}_{date_now}"
    PRETRAINED_MODEL_PATH = os.path.join(model_dir, "toy_llm_unified_pretrained.pth")
    FINETUNED_MODEL_PATH = os.path.join(model_dir, "toy_llm_qasrl_finetuned.pth")

    os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
    print(f"Best pre-trained model will be saved to: {PRETRAINED_MODEL_PATH}")

    if not SKIP_PRETRAIN:
        print(f"\n--- Starting CLM Pre-training on nq_open for '{model_config.name}' model ---")
        pretrain_model = ToyLLMForPretraining(base_llm).to(DEVICE)
        
        # Use IterableDataset for streaming data
        train_dataset_clm = NQOpenDataset(split="train")
        val_dataset_clm = NQOpenDataset(split="validation")

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
            pretrain_model.parameters(), lr=train_params['pretrain_lr'], weight_decay=0.1
        )
        scheduler_pretrain = ReduceLROnPlateau(
            optimizer_pretrain,
            mode="min",
            factor=0.5,
            patience=2,
        )
        criterion_clm = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        global_step = 0

        # Get gradient accumulation steps for pre-training (default to 1)
        pretrain_grad_accum = train_params.get('gradient_accumulation_steps', 1)
        effective_pretrain_batch = train_params['batch_size_pretrain'] * pretrain_grad_accum
        print(f"Pre-training batch size: {train_params['batch_size_pretrain']} | Gradient accumulation: {pretrain_grad_accum} | Effective batch size: {effective_pretrain_batch}")

        for epoch in range(1, train_params['num_pretrain_epochs'] + 1):
            pretrain_model.train()
            total_train_loss = 0
            optimizer_pretrain.zero_grad()

            # The DataLoader for an IterableDataset will handle shuffling internally
            # or in this case, iterate through the entire stream.
            for batch_idx, (clm_inputs, clm_labels, attention_masks) in enumerate(
                train_dataloader_clm
            ):
                global_step += 1
                if clm_inputs is None:
                    continue

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

                # Scale loss for gradient accumulation
                loss = loss / pretrain_grad_accum
                loss.backward()
                total_train_loss += loss.item() * pretrain_grad_accum

                # Only update weights after accumulating gradients
                if (batch_idx + 1) % pretrain_grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 0.5)
                    optimizer_pretrain.step()
                    optimizer_pretrain.zero_grad()

            avg_train_loss = total_train_loss / (batch_idx + 1)

            avg_val_loss = validate_pretrain_epoch(
                pretrain_model, val_dataloader_clm, criterion_clm, DEVICE
            )
            current_lr = optimizer_pretrain.param_groups[0]["lr"]
            print(
                f"--- End of Pre-train Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Validation Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f} ---"
            )

            scheduler_pretrain.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(pretrain_model.state_dict(), PRETRAINED_MODEL_PATH)
                print(
                    f"Validation loss improved. Saving model to {PRETRAINED_MODEL_PATH}"
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= train_params['pretrain_patience']:
                print(
                    f"\nEarly stopping triggered after {train_params['pretrain_patience']} epochs without improvement."
                )
                break

        print(f"\nLoading best pretrained model from {PRETRAINED_MODEL_PATH}")
        pretrain_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

        def simple_generate(
            model, prompt, tokenizer, max_len=20, top_p=0.9, device=DEVICE
        ):
            model.eval()
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)

            for _ in range(max_len):
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                with torch.no_grad():
                    logits = model(
                        input_ids, attention_mask=attention_mask, is_causal=True
                    )
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float("inf")
                next_token_id = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), num_samples=1
                ).unsqueeze(0)

                if next_token_id.item() == tokenizer.sep_token_id:
                    break
                input_ids = torch.cat([input_ids, next_token_id], dim=1)

            return tokenizer.decode(input_ids[0], skip_special_tokens=True)

        test_prompt = "Question: where was the statue of liberty originally built"
        print("\nTesting prompt completion after pretraining:")
        print(f"Prompt: '{test_prompt}'")
        completion = simple_generate(
            pretrain_model, test_prompt, tokenizer, max_len=10, device=DEVICE
        )
        print(f"Model completion: {completion}")
    else:
        print(
            f"\nSkipping pre-training. Attempting to load model from: {PRETRAINED_MODEL_PATH}"
        )

    if not SKIP_FINETUNE:
        print(f"\n--- Starting Fine-tuning on QA-SRL for '{model_config.name}' model ---")
        qa_model = ToyLLMForQuestionAnswering(base_llm).to(DEVICE)
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
            print(f"WARNING: Pre-trained model not found at {PRETRAINED_MODEL_PATH}.")
            print("Fine-tuning will start from randomly initialized weights.")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Fine-tuning will start from randomly initialized weights.")

        qa_train_dataset = QASRLDataset(
            split="train", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
        )
        qa_train_dataloader = DataLoader(
            qa_train_dataset,
            batch_size=train_params['batch_size_qa'],
            shuffle=True,
            collate_fn=collate_batch_qa,
        )
        
        qa_val_dataset = QASRLDataset(
            split="validation", tokenizer=tokenizer, max_seq_len=train_params['max_seq_len']
        )
        qa_val_dataloader = DataLoader(
            qa_val_dataset,
            batch_size=train_params['batch_size_qa'],
            shuffle=False,
            collate_fn=collate_batch_qa,
        )

        optimizer_finetune = optim.AdamW(qa_model.parameters(), lr=train_params['finetune_lr'],  weight_decay=0.01)
        scheduler_finetune = ReduceLROnPlateau(
            optimizer_finetune, mode="min", factor=0.5, patience=2
        )

        # Get gradient accumulation steps (default to 1 if not specified)
        gradient_accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        finetune_patience = train_params.get('finetune_patience', 5)

        effective_batch_size = train_params['batch_size_qa'] * gradient_accumulation_steps
        print(f"Starting fine-tuning for up to {train_params['num_finetune_epochs']} epochs...")
        print(f"Batch size: {train_params['batch_size_qa']} | Gradient accumulation steps: {gradient_accumulation_steps} | Effective batch size: {effective_batch_size}")
        print(f"Early stopping patience: {finetune_patience} epochs")

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        final_train_loss = 0.0
        epochs_no_improve = 0

        for epoch_num in range(1, train_params['num_finetune_epochs'] + 1):
            avg_train_loss = finetune_qa_epoch(
                qa_model, qa_train_dataloader, optimizer_finetune, DEVICE, epoch_num,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            final_train_loss = avg_train_loss

            avg_val_loss, val_acc, val_f1 = validate_qa_epoch(qa_model, qa_val_dataloader, DEVICE)

            scheduler_finetune.step(avg_val_loss)

            current_lr = optimizer_finetune.param_groups[0]['lr']
            print(
                f"--- End of Finetune Epoch {epoch_num} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
                f"| LR: {current_lr:.6f} ---"
            )

            # Track best model by validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_f1 = val_f1
                epochs_no_improve = 0
                torch.save(qa_model.state_dict(), FINETUNED_MODEL_PATH)
                print(f"Validation loss improved. Saving best model to {FINETUNED_MODEL_PATH}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")

            # Early stopping check
            if epochs_no_improve >= finetune_patience:
                print(f"\nEarly stopping triggered after {finetune_patience} epochs without improvement.")
                print(f"Best validation loss: {best_val_loss:.4f} | Best F1: {best_val_f1:.4f}")
                break

        print("\nFine-tuning finished.")
        print(f"Best fine-tuned QA model state_dict saved to {FINETUNED_MODEL_PATH}")

        print(f"Loading best model from {FINETUNED_MODEL_PATH} for final stats.")
        qa_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH))

        final_val_acc, final_val_f1 = evaluate_qa_metrics(qa_model, qa_val_dataloader, DEVICE)
        print(
            f"Final Validation Accuracy: {final_val_acc:.4f} | Final F1: {final_val_f1:.4f}"
        )
        stats = {
            "pretrain_epochs": "skipped" if SKIP_PRETRAIN else epoch,
            "finetune_epochs": train_params['num_finetune_epochs'],
            "final_train_loss": final_train_loss,
            "best_validation_loss": best_val_loss,
            "final_validation_accuracy": final_val_acc,
            "final_validation_f1": final_val_f1,
        }
        stats_path = os.path.join(model_dir, "finetune_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Final training statistics saved to '{stats_path}'")
    else:
        print("Skipping fine-tuning step.")