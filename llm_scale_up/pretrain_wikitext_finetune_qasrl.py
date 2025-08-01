import json
import os
import time
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

try:
    from llm_scale_up.utils.eval_metrics import evaluate_qa_metrics
except ImportError:
    print("Warning: `evaluate_qa_metrics` not found. Using a mock function.")

    def evaluate_qa_metrics(model, dataloader, device):
        # Mock function returns dummy values if the original is not available
        print("Mock evaluation: Returning 0.0 for accuracy and F1.")
        return 0.0, 0.0


# need to modularise this later


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

        # Apply causal mask if requested
        if is_causal:
            causal_mask = torch.triu(torch.ones_like(att_scores), diagonal=1).bool()
            att_scores = att_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # This unsqueeze is for the padding mask
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att_scores = att_scores.masked_fill(attention_mask == 0, float("-inf"))

        att_probs = torch.softmax(att_scores, dim=-1)
        output = (att_probs @ v_h).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class ToyFeedForward(nn.Module):
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
            padding_idx = module.padding_idx
            temp_weight = torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if padding_idx is not None:
                with torch.no_grad():
                    temp_weight[padding_idx].fill_(0)
            module.weight.data = temp_weight

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
    # Input is the sequence, except for the last token
    inputs = token_ids[:, :-1].contiguous()
    # Use .clone() to create a new copy of the data for labels
    labels = token_ids[:, 1:].clone().contiguous()

    # This modification now only affects the 'labels' tensor and won't corrupt 'inputs'
    labels[labels == pad_token_id] = -100
    return inputs, labels


def collate_batch_clm(batch, tokenizer, max_seq_len, device):
    """Collates a batch of text for Causal Language Modeling using a Hugging Face tokenizer."""
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

    # Use the create_clm_inputs_and_labels function to prepare model inputs
    clm_inputs, clm_labels = create_clm_inputs_and_labels(
        tokenized_texts["input_ids"], tokenizer.pad_token_id
    )

    # The attention mask needs to correspond to the `clm_inputs`, so we slice it.
    attention_mask = tokenized_texts["attention_mask"][:, :-1].contiguous()

    return (
        clm_inputs.to(device),
        clm_labels.to(device),
        attention_mask.to(device),
    )


def validate_pretrain_epoch(model, dataloader, criterion, device):
    """Runs a validation loop for one epoch during pre-training."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
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


class WikiText2Dataset:
    DATASET_NAME = "Salesforce/wikitext"
    CONFIG_NAME = "wikitext-2-v1"

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _hf_dataset_to_line_iterator(self, hf_split_dataset) -> Iterator[str]:
        for item in hf_split_dataset:
            text_line = item.get("text", "").strip()
            if (
                text_line
                and not text_line.startswith(" = ")
                and not text_line.startswith("= ")
            ):
                yield text_line

    def __call__(self) -> Tuple[Iterator[str], Iterator[str], Iterator[str]]:
        train_hf_ds = load_dataset(
            self.DATASET_NAME, self.CONFIG_NAME, split="train", cache_dir=self.cache_dir
        )
        valid_hf_ds = load_dataset(
            self.DATASET_NAME,
            self.CONFIG_NAME,
            split="validation",
            cache_dir=self.cache_dir,
        )
        test_hf_ds = load_dataset(
            self.DATASET_NAME, self.CONFIG_NAME, split="test", cache_dir=self.cache_dir
        )
        return (
            self._hf_dataset_to_line_iterator(train_hf_ds),
            self._hf_dataset_to_line_iterator(valid_hf_ds),
            self._hf_dataset_to_line_iterator(test_hf_ds),
        )


class QASRLDataset(Dataset):
    def __init__(self, split, tokenizer, max_seq_len):
        print(f"Loading and processing qa_srl dataset for '{split}' split...")
        self.dataset = load_dataset("qa_srl", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.processed_data = self._preprocess()

    def _preprocess(self):
        processed = []
        for example in self.dataset:
            context = example["sentence"]
            # Reconstruct question, skipping placeholder "_"
            question = " ".join(
                [token for token in example["question"] if token != "_"]
            )

            # Some answers might be empty or invalid, skip them
            if not example.get("answers"):
                continue

            for answer_text in example["answers"]:
                # Find the character start and end positions of the answer
                char_start = context.lower().find(answer_text.lower())
                if char_start == -1:
                    continue
                char_end = char_start + len(answer_text)

                # Tokenize the context and question together
                encoding = self.tokenizer(
                    question,
                    context,
                    truncation="only_second",  # Truncate context if needed
                    max_length=self.max_seq_len,
                    stride=128,  # Allow for overlapping context
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )

                # Find the token indices that correspond to the answer span
                for i in range(len(encoding["input_ids"])):
                    sequence_ids = encoding.sequence_ids(i)
                    # The context is the second part of the sequence (id 1)
                    context_indices = [
                        idx for idx, sid in enumerate(sequence_ids) if sid == 1
                    ]

                    if not context_indices:
                        continue

                    # Get the start and end character positions for each token in the context
                    offset_mapping = encoding["offset_mapping"][i]
                    context_offsets = [offset_mapping[j] for j in context_indices]

                    token_start_index = -1
                    token_end_index = -1

                    # Find the start and end tokens of our answer
                    for idx, (start, end) in zip(context_indices, context_offsets):
                        if start <= char_start < end:
                            token_start_index = idx
                        if start < char_end <= end:
                            token_end_index = idx

                    # If we found a valid span
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
                        # We only add the first valid span found for this answer
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


def finetune_qa_epoch(model, dataloader, optimizer, device, epoch_num, log_interval=50):
    model.train()
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, start_pos, end_pos) in enumerate(
        dataloader
    ):
        input_ids, attention_mask, start_pos, end_pos = (
            input_ids.to(device),
            attention_mask.to(device),
            start_pos.to(device),
            end_pos.to(device),
        )
        optimizer.zero_grad()
        loss, _, _ = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_pos,
            end_positions=end_pos,
        )
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Finetune Epoch {epoch_num} | Batch {batch_idx}/{len(dataloader)} | Avg Loss: {avg_loss:.4f}"
            )
    return total_loss / len(dataloader)


if __name__ == "__main__":
    SKIP_PRETRAIN = False
    SKIP_FINETUNE = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Use a Standard Tokenizer ---
    print("Loading standard tokenizer ('bert-base-uncased')...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    VOCAB_SIZE = tokenizer.vocab_size
    PAD_TOKEN_ID = tokenizer.pad_token_id
    print(f"Standard vocabulary size: {VOCAB_SIZE}")
    print(f"PAD token ID: {PAD_TOKEN_ID}")

    # --- Model and Training Config ---
    tiny_config = LLMConfig(name="TinyQA", n_layers=2, hidden_size=128, n_heads=2)
    MAX_SEQ_LEN = 256
    DROPOUT_RATE = 0.1
    PRETRAIN_LR = 3e-4
    NUM_PRETRAIN_EPOCHS = 100  # Set a high number, early stopping will find the best
    PRETRAIN_PATIENCE = 5  # Early stopping patience
    WARMUP_STEPS = 500  # Number of warm-up steps for the learning rate

    base_llm = ToyLLM(
        config=tiny_config,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        pad_token_id=PAD_TOKEN_ID,
        dropout_rate=DROPOUT_RATE,
    )

    num_params = sum(p.numel() for p in base_llm.parameters() if p.requires_grad)
    print(
        f"Instantiated Base Model: {tiny_config.name} with {num_params:,} trainable parameters."
    )

    date_now = time.strftime("%Y%m%d-%H%M%S")
    NUM_FINETUNE_EPOCHS = 3
    PRETRAINED_MODEL_PATH = f"models/date_{date_now}/toy_llm_unified_pretrained.pth"
    FINETUNED_MODEL_PATH = f"models/date_{date_now}/toy_llm_qasrl_finetuned.pth"

    os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
    print(f"Best pre-trained model will be saved to: {PRETRAINED_MODEL_PATH}")

    if not SKIP_PRETRAIN:
        print("\n--- Starting CLM Pre-training ---")
        pretrain_model = ToyLLMForPretraining(base_llm).to(DEVICE)

        dataset_loader_main = WikiText2Dataset()
        train_iter_raw, valid_iter_raw, _ = dataset_loader_main()

        # Filter out empty lines
        train_data_list = [line for line in train_iter_raw if line.strip()]
        valid_data_list = [line for line in valid_iter_raw if line.strip()]

        BATCH_SIZE_PRETRAIN = 16
        collate_fn_clm = lambda batch: collate_batch_clm(
            batch, tokenizer, MAX_SEQ_LEN, DEVICE
        )
        train_dataloader_clm = DataLoader(
            train_data_list,
            batch_size=BATCH_SIZE_PRETRAIN,
            shuffle=True,
            collate_fn=collate_fn_clm,
        )
        val_dataloader_clm = DataLoader(
            valid_data_list,
            batch_size=BATCH_SIZE_PRETRAIN,
            shuffle=False,
            collate_fn=collate_fn_clm,
        )

        optimizer_pretrain = optim.AdamW(
            pretrain_model.parameters(), lr=PRETRAIN_LR, weight_decay=0.1
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

        for epoch in range(1, NUM_PRETRAIN_EPOCHS + 1):
            pretrain_model.train()
            total_train_loss = 0

            for batch_idx, (clm_inputs, clm_labels, attention_masks) in enumerate(
                train_dataloader_clm
            ):
                global_step += 1
                if clm_inputs is None:
                    continue

                # --- LR Warm-up Logic ---
                if global_step < WARMUP_STEPS:
                    lr_scale = global_step / WARMUP_STEPS
                    for param_group in optimizer_pretrain.param_groups:
                        param_group["lr"] = lr_scale * PRETRAIN_LR

                optimizer_pretrain.zero_grad()
                output_logits = pretrain_model(
                    clm_inputs, attention_mask=attention_masks, is_causal=True
                )
                loss = criterion_clm(
                    output_logits.view(-1, pretrain_model.llm.vocab_size),
                    clm_labels.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), 0.5)
                optimizer_pretrain.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader_clm)
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

            if epochs_no_improve >= PRETRAIN_PATIENCE:
                print(
                    f"\nEarly stopping triggered after {PRETRAIN_PATIENCE} epochs without improvement."
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

        test_prompt = "The capital of France is"
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
        print("\n--- Starting Fine-tuning on QA-SRL ---")
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

        BATCH_SIZE_QA = 8
        qa_train_dataset = QASRLDataset(
            split="train", tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN
        )
        qa_train_dataloader = DataLoader(
            qa_train_dataset,
            batch_size=BATCH_SIZE_QA,
            shuffle=True,
            collate_fn=collate_batch_qa,
        )

        optimizer_finetune = optim.AdamW(qa_model.parameters(), lr=5e-5)
        scheduler_finetune = ReduceLROnPlateau(
            optimizer_finetune, mode="min", factor=0.5, patience=2
        )

        print(f"Starting fine-tuning for {NUM_FINETUNE_EPOCHS} epochs...")
        avg_epoch_loss = 0.0
        for epoch_num in range(1, NUM_FINETUNE_EPOCHS + 1):
            avg_epoch_loss = finetune_qa_epoch(
                qa_model, qa_train_dataloader, optimizer_finetune, DEVICE, epoch_num
            )
            scheduler_finetune.step(avg_epoch_loss)
            print(
                f"--- End of Finetune Epoch {epoch_num} | Average QA Loss: {avg_epoch_loss:.4f} "
                f"| LR: {scheduler_finetune.optimizer.param_groups[0]['lr']:.6f} ---"
            )

        print("\nFine-tuning finished.")
        torch.save(qa_model.state_dict(), FINETUNED_MODEL_PATH)
        print(f"Fine-tuned QA model state_dict saved to {FINETUNED_MODEL_PATH}")

        # The mock evaluate_qa_metrics will run here if the original is not found
        final_acc, final_f1 = evaluate_qa_metrics(qa_model, qa_train_dataloader, DEVICE)
        print(
            f"Final Training Set Accuracy: {final_acc:.4f} | Final F1: {final_f1:.4f}"
        )
        stats = {
            "pretrain_epochs": "skipped" if SKIP_PRETRAIN else epoch,
            "finetune_epochs": NUM_FINETUNE_EPOCHS,
            "final_loss": avg_epoch_loss,
            "final_accuracy": final_acc,
            "final_f1": final_f1,
        }
        stats_path = f"models/date_{date_now}/finetune_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Final training statistics saved to '{stats_path}'")
    else:
        print("Skipping fine-tuning step.")

    print("\n--- Running Validation Metrics on QA-SRL Validation Set ---")
    qa_model_for_eval = ToyLLMForQuestionAnswering(base_llm).to(DEVICE)
    try:
        if os.path.exists(FINETUNED_MODEL_PATH):
            qa_model_for_eval.load_state_dict(
                torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
            )
            print(f"Loaded finetuned model from {FINETUNED_MODEL_PATH} for validation.")
        elif os.path.exists(PRETRAINED_MODEL_PATH):
            pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
            qa_model_for_eval.llm.load_state_dict(
                {
                    k.replace("llm.", ""): v
                    for k, v in pretrained_dict.items()
                    if k.startswith("llm.")
                }
            )
            print(
                f"Loaded pretrained model from {PRETRAINED_MODEL_PATH} for validation."
            )
        else:
            print(
                "No trained weights found. Using randomly initialized model for validation."
            )
    except Exception as e:
        print(f"Error loading model weights for validation: {e}. Using random weights.")

    BATCH_SIZE_QA = 8
    qa_val_dataset = QASRLDataset(
        split="validation", tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN
    )
    qa_val_dataloader = DataLoader(
        qa_val_dataset,
        batch_size=BATCH_SIZE_QA,
        shuffle=False,
        collate_fn=collate_batch_qa,
    )

    val_acc, val_f1 = evaluate_qa_metrics(qa_model_for_eval, qa_val_dataloader, DEVICE)
    print(
        f"Final Validation Accuracy: {val_acc:.4f} | Final Validation F1: {val_f1:.4f}"
    )
    val_stats_path = f"models/date_{date_now}/validation_stats.json"
    with open(val_stats_path, "w") as f:
        json.dump(
            {
                "final_validation_accuracy": val_acc,
                "final_validation_f1": val_f1,
            },
            f,
            indent=2,
        )
    print(f"Validation statistics saved to '{val_stats_path}'")
