import json
import os
import re
import time
from itertools import chain
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from llm_scale_up.utils.eval_metrics import evaluate_qa_metrics

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

    def forward(self, q, k, v, attention_mask=None):
        B, T, C = q.shape
        q_h = self.q_proj(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(v).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att_scores = (q_h @ k_h.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
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

    def forward(self, x, attention_mask=None):
        normed_x = self.norm1(x)
        attn_output = self.attention(
            normed_x, normed_x, normed_x, attention_mask=attention_mask
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

    def forward(self, input_ids, attention_mask=None):
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
            x = layer(x, attention_mask=attention_mask)

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
        sequence_output = self.llm(input_ids, attention_mask=attention_mask)
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
    """Wrapper for the pre-training phase, includes the MLM head."""

    def __init__(self, base_model: ToyLLM):
        super().__init__()
        self.llm = base_model
        self.mlm_output_head = nn.Linear(
            self.llm.config.hidden_size, self.llm.vocab_size
        )

    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.llm(input_ids, attention_mask=attention_mask)
        logits = self.mlm_output_head(sequence_output)
        return logits


def basic_english_tokenizer(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return tokens


TOKENIZER = basic_english_tokenizer
UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN = (
    "<unk>",
    "<pad>",
    "<mask>",
    "<bos>",
    "<eos>",
    "<sep>",
)
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]


def yield_tokens(data_iter, tokenizer_fn):
    for text in data_iter:
        yield tokenizer_fn(text)


def build_vocab_from_iterator(iterator, min_freq, specials):
    from collections import Counter, OrderedDict

    token_counts = Counter()
    for tokens in iterator:
        token_counts.update(tokens)
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: (x[1], x[0]), reverse=True
    )
    filtered_tokens = [
        token for token, count in sorted_by_freq_tuples if count >= min_freq
    ]
    vocab_list = OrderedDict()
    for token in specials:
        vocab_list[token] = len(vocab_list)
    for token in filtered_tokens:
        if token not in vocab_list:
            vocab_list[token] = len(vocab_list)
    token_to_index = {token: idx for idx, token in enumerate(vocab_list.keys())}

    class Vocab:
        def __init__(self, token_to_index, specials):
            self.token_to_index = token_to_index
            self.index_to_token = {idx: token for token, idx in token_to_index.items()}
            self.specials = specials
            self.unk_index = token_to_index.get("<unk>")
            self.default_index = self.unk_index

        def __len__(self):
            return len(self.token_to_index)

        def __getitem__(self, token):
            return self.token_to_index.get(token, self.default_index)

        def set_default_index(self, index):
            self.default_index = index

    return Vocab(token_to_index, specials)


def build_unified_vocab(min_freq=1):
    """Builds a vocabulary from both wikitext-2 and qa_srl train and validation splits."""
    print(
        "Building unified vocabulary from wikitext-2 and qa_srl (train + validation)..."
    )

    wikitext_loader = WikiText2Dataset()
    wikitext_train_iter, _, _ = wikitext_loader()
    wikitext_tokens_iter = yield_tokens(wikitext_train_iter, TOKENIZER)

    print("Loading qa_srl 'train' and 'validation' splits for vocabulary building...")
    qa_srl_train = load_dataset("qa_srl", split="train", trust_remote_code=True)
    qa_srl_val = load_dataset("qa_srl", split="validation", trust_remote_code=True)

    def qa_srl_token_iterator():
        for example in chain(qa_srl_train, qa_srl_val):
            # Yield tokens from the sentence
            yield TOKENIZER(example["sentence"])
            # Yield tokens from the question
            if "question" in example:
                if isinstance(example["question"], list):
                    # Join the question tokens if they're in a list
                    yield TOKENIZER(" ".join(example["question"]))
                else:
                    yield TOKENIZER(example["question"])

    qa_srl_tokens_iter = qa_srl_token_iterator()

    # Chain the iterators together
    combined_iterator = chain(wikitext_tokens_iter, qa_srl_tokens_iter)

    vocab = build_vocab_from_iterator(
        combined_iterator,
        min_freq=min_freq,
        specials=SPECIAL_TOKENS,
    )
    print("Unified vocabulary built successfully.")
    return vocab


def create_mlm_inputs_and_labels(token_ids, vocab, mlm_probability=0.15):
    mask_token_id = vocab[MASK_TOKEN]
    pad_token_id = vocab[PAD_TOKEN]
    inputs = token_ids.clone()
    labels = token_ids.clone()
    prob_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = (
        (labels == pad_token_id)
        | (labels == vocab[BOS_TOKEN])
        | (labels == vocab[EOS_TOKEN])
        | (labels == mask_token_id)
    )
    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    inputs[masked_indices] = mask_token_id
    labels[~masked_indices] = -100
    return inputs, labels


def collate_batch_mlm(batch, vocab, tokenizer_fn, max_seq_len, device):
    bos_token_id, eos_token_id, pad_token_id = (
        vocab[BOS_TOKEN],
        vocab[EOS_TOKEN],
        vocab[PAD_TOKEN],
    )
    processed_texts = []
    for text_sample in batch:
        if not text_sample.strip():
            continue
        tokens = (
            [bos_token_id]
            + [vocab[token] for token in tokenizer_fn(text_sample)]
            + [eos_token_id]
        )
        processed_texts.append(torch.tensor(tokens, dtype=torch.long))
    if not processed_texts:
        return None, None, None
    padded_sequences = pad_sequence(
        [p[:max_seq_len] for p in processed_texts],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_masks = (padded_sequences != pad_token_id).long()
    mlm_inputs, mlm_labels = create_mlm_inputs_and_labels(padded_sequences, vocab)
    return mlm_inputs.to(device), mlm_labels.to(device), attention_masks.to(device)


def pretrain_epoch(
    model, dataloader, optimizer, criterion, device, epoch_num, log_interval=100
):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_idx, (mlm_inputs, mlm_labels, attention_masks) in enumerate(dataloader):
        if mlm_inputs is None:
            continue
        optimizer.zero_grad()
        output_logits = model(mlm_inputs, attention_mask=attention_masks)
        loss = criterion(
            output_logits.view(-1, model.llm.vocab_size), mlm_labels.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Pre-train Epoch {epoch_num} | Batch {batch_idx}/{len(dataloader)} | Avg Loss: {avg_loss:.4f}"
            )
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
    def __init__(self, split, tokenizer_fn, vocab, max_seq_len):
        print(f"Loading qa_srl dataset for '{split}' split...")
        self.dataset = load_dataset("qa_srl", split=split, trust_remote_code=True)
        self.tokenizer_fn = tokenizer_fn
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.bos_id, self.sep_id, self.eos_id, self.pad_id = (
            vocab[BOS_TOKEN],
            vocab[SEP_TOKEN],
            vocab[EOS_TOKEN],
            vocab[PAD_TOKEN],
        )
        self.processed_data = self._preprocess()

    def _preprocess(self):
        processed = []
        for example in self.dataset:
            context = example["sentence"]
            context_tokens = self.tokenizer_fn(context)

            question = " ".join(
                [token for token in example["question"] if token != "_"]
            )
            question_tokens = self.tokenizer_fn(question)

            input_tokens = (
                [self.vocab[t] for t in question_tokens]
                + [self.sep_id]
                + [self.vocab[t] for t in context_tokens]
            )
            input_ids = (
                [self.bos_id] + input_tokens[: self.max_seq_len - 2] + [self.eos_id]
            )

            for answer in example["answers"]:
                answer_tokens = self.tokenizer_fn(answer)
                context_offset = len(question_tokens) + 1
                answer_ids = [self.vocab[t] for t in answer_tokens]
                context_ids = [self.vocab[t] for t in context_tokens]

                start_pos, end_pos = 0, 0
                for i in range(len(context_ids) - len(answer_ids) + 1):
                    if context_ids[i : i + len(answer_ids)] == answer_ids:
                        start_pos = context_offset + i + 1
                        end_pos = start_pos + len(answer_ids) - 1
                        break

                if start_pos > 0 and end_pos < len(input_ids) - 1:
                    processed.append(
                        {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "start_position": torch.tensor(start_pos, dtype=torch.long),
                            "end_position": torch.tensor(end_pos, dtype=torch.long),
                        }
                    )

        print(
            f"Finished preprocessing '{self.dataset.split}'. Found {len(processed)} valid QA examples."
        )
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_batch_qa(batch, pad_token_id):
    input_ids_list = [item["input_ids"] for item in batch]
    start_positions = torch.stack([item["start_position"] for item in batch])
    end_positions = torch.stack([item["end_position"] for item in batch])
    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = (padded_input_ids != pad_token_id).long()
    return padded_input_ids, attention_mask, start_positions, end_positions


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
    # Set these to True to skip steps
    SKIP_PRETRAIN = False
    SKIP_FINETUNE = False

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Build a single UNIFIED vocab from both datasets
    vocab = build_unified_vocab(min_freq=5)
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN_ID = vocab[PAD_TOKEN]
    print(f"Unified vocabulary size: {VOCAB_SIZE}")
    print(f"PAD token ID: {PAD_TOKEN_ID}")

    tiny_config = LLMConfig(name="TinyQA", n_layers=2, hidden_size=128, n_heads=2)
    MAX_SEQ_LEN = 256
    DROPOUT_RATE = 0.1

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
    NUM_FINETUNE_EPOCHS = 30
    PRETRAINED_MODEL_PATH = f"models/date_{date_now}/toy_llm_unified_pretrained.pth"
    FINETUNED_MODEL_PATH = f"models/date_{date_now}/toy_llm_qasrl_finetuned.pth"

    os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FINETUNED_MODEL_PATH), exist_ok=True)
    print(f"Pre-trained model will be saved to created dir: {PRETRAINED_MODEL_PATH}")

    if not SKIP_PRETRAIN:
        print("\n--- Starting MLM Pre-training with Unified Vocab ---")
        pretrain_model = ToyLLMForPretraining(base_llm).to(DEVICE)

        # We only pre-train on wikitext-2, but with the full, combined vocabulary
        dataset_loader_main = WikiText2Dataset()
        train_iter_raw, _, _ = dataset_loader_main()
        train_data_list = [line for line in train_iter_raw if line.strip()]

        BATCH_SIZE_PRETRAIN = 16
        collate_fn_mlm = lambda batch: collate_batch_mlm(
            batch, vocab, TOKENIZER, MAX_SEQ_LEN, DEVICE
        )
        train_dataloader_mlm = DataLoader(
            train_data_list,
            batch_size=BATCH_SIZE_PRETRAIN,
            shuffle=True,
            collate_fn=collate_fn_mlm,
        )

        optimizer_pretrain = optim.AdamW(pretrain_model.parameters(), lr=1e-4)
        criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        NUM_PRETRAIN_EPOCHS = 50

        for epoch in range(1, NUM_PRETRAIN_EPOCHS + 1):
            avg_epoch_loss = pretrain_epoch(
                pretrain_model,
                train_dataloader_mlm,
                optimizer_pretrain,
                criterion_mlm,
                DEVICE,
                epoch,
                log_interval=500,
            )
            print(
                f"--- End of Pre-train Epoch {epoch} | Average Loss: {avg_epoch_loss:.4f} ---"
            )

        torch.save(pretrain_model.llm.state_dict(), PRETRAINED_MODEL_PATH)
        print(f"Pre-trained base model state_dict saved to {PRETRAINED_MODEL_PATH}")
    else:
        print(
            f"\nSkipping pre-training. Attempting to load model from: {PRETRAINED_MODEL_PATH}"
        )

    if not SKIP_FINETUNE:
        print("\n--- Starting Fine-tuning on QA-SRL ---")

        qa_model = ToyLLMForQuestionAnswering(base_llm).to(DEVICE)

        try:
            pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
            qa_model.llm.load_state_dict(pretrained_dict)
            print("Successfully loaded pre-trained weights into the QA model.")
        except FileNotFoundError:
            print(f"WARNING: Pre-trained model not found at {PRETRAINED_MODEL_PATH}.")
            print("Fine-tuning will start from randomly initialized weights.")

        BATCH_SIZE_QA = 8
        qa_train_dataset = QASRLDataset(
            split="train", tokenizer_fn=TOKENIZER, vocab=vocab, max_seq_len=MAX_SEQ_LEN
        )
        collate_fn_qa = lambda batch: collate_batch_qa(batch, PAD_TOKEN_ID)
        qa_train_dataloader = DataLoader(
            qa_train_dataset,
            batch_size=BATCH_SIZE_QA,
            shuffle=True,
            collate_fn=collate_fn_qa,
        )

        optimizer_finetune = optim.AdamW(qa_model.parameters(), lr=5e-5)

        print(f"Starting fine-tuning for {NUM_FINETUNE_EPOCHS} epochs...")
        for epoch in range(1, NUM_FINETUNE_EPOCHS + 1):
            avg_epoch_loss = finetune_qa_epoch(
                qa_model, qa_train_dataloader, optimizer_finetune, DEVICE, epoch
            )
            print(
                f"--- End of Finetune Epoch {epoch} | Average QA Loss: {avg_epoch_loss:.4f} ---"
            )

        print("\nFine-tuning finished.")
        torch.save(qa_model.state_dict(), FINETUNED_MODEL_PATH)
        print(f"Fine-tuned QA model state_dict saved to {FINETUNED_MODEL_PATH}")

        final_acc, final_f1 = evaluate_qa_metrics(qa_model, qa_train_dataloader, DEVICE)
        print(f"Final Accuracy: {final_acc:.4f} | Final F1: {final_f1:.4f}")

        stats = {
            "pretrain_epochs": 0 if SKIP_PRETRAIN else NUM_PRETRAIN_EPOCHS,
            "finetune_epochs": NUM_FINETUNE_EPOCHS,
            "final_loss": avg_epoch_loss,
            "final_accuracy": final_acc,
            "final_f1": final_f1,
        }

        with open("models/finetune_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("Final training statistics saved to 'models/finetune_stats.json'")

    else:
        print("Skipping fine-tuning step.")

    # Always run validation, regardless of SKIP_PRETRAIN or SKIP_FINETUNE
    print("\n--- Running Validation Metrics on QA-SRL Validation Set ---")
    qa_model = ToyLLMForQuestionAnswering(base_llm).to(DEVICE)
    try:
        # Try to load finetuned weights first, else pretrained, else random
        if os.path.exists(FINETUNED_MODEL_PATH):
            qa_model.load_state_dict(
                torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
            )
            print(f"Loaded finetuned model from {FINETUNED_MODEL_PATH}")
        elif os.path.exists(PRETRAINED_MODEL_PATH):
            pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
            qa_model.llm.load_state_dict(pretrained_dict)
            print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")
        else:
            print(
                "No trained weights found. Using randomly initialized model for validation."
            )
    except Exception as e:
        print(
            f"Error loading model weights: {e}\nUsing randomly initialized model for validation."
        )

    BATCH_SIZE_QA = 8
    qa_val_dataset = QASRLDataset(
        split="validation", tokenizer_fn=TOKENIZER, vocab=vocab, max_seq_len=MAX_SEQ_LEN
    )
    collate_fn_qa = lambda batch: collate_batch_qa(batch, PAD_TOKEN_ID)
    qa_val_dataloader = DataLoader(
        qa_val_dataset,
        batch_size=BATCH_SIZE_QA,
        shuffle=False,
        collate_fn=collate_fn_qa,
    )

    final_acc, final_f1 = evaluate_qa_metrics(qa_model, qa_val_dataloader, DEVICE)
    print(
        f"Final Validation Accuracy: {final_acc:.4f} | Final Validation F1: {final_f1:.4f}"
    )
    with open("models/validation_stats.json", "w") as f:
        json.dump(
            {
                "final_validation_accuracy": final_acc,
                "final_validation_f1": final_f1,
            },
            f,
            indent=2,
        )

    # Print out an example train and validation example at the end
    print("\nExample from QA-SRL train set:")
    qa_train_dataset = QASRLDataset(
        split="train", tokenizer_fn=TOKENIZER, vocab=vocab, max_seq_len=MAX_SEQ_LEN
    )

    def decode_input_ids(input_ids, vocab):
        idx_to_token = vocab.index_to_token
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        return [idx_to_token[idx] for idx in input_ids]

    if len(qa_train_dataset) > 0:
        train_ex = qa_train_dataset[0]
        print(train_ex)
        train_tokens = decode_input_ids(train_ex["input_ids"], vocab)
        if "<sep>" in train_tokens:
            sep_idx = train_tokens.index("<sep>")
            question_tokens = train_tokens[1:sep_idx]  # skip <bos>
            context_tokens = train_tokens[sep_idx + 1 : -1]  # skip <eos>
            print("Train question:", " ".join(question_tokens))
            print("Train context:", " ".join(context_tokens))
            # Print answer span
            start = train_ex["start_position"].item()
            end = train_ex["end_position"].item()
            answer_tokens = train_tokens[start : end + 1]
            print("Train answer:", " ".join(answer_tokens))
        else:
            print("Could not find <sep> token in train input_ids.")
    else:
        print("No examples in train set.")

    print("\nExample from QA-SRL validation set:")
    if len(qa_val_dataset) > 0:
        val_ex = qa_val_dataset[0]
        print(val_ex)
        val_tokens = decode_input_ids(val_ex["input_ids"], vocab)
        if "<sep>" in val_tokens:
            sep_idx = val_tokens.index("<sep>")
            question_tokens = val_tokens[1:sep_idx]
            context_tokens = val_tokens[sep_idx + 1 : -1]
            print("Validation question:", " ".join(question_tokens))
            print("Validation context:", " ".join(context_tokens))
            # Print answer span
            start = val_ex["start_position"].item()
            end = val_ex["end_position"].item()
            answer_tokens = val_tokens[start : end + 1]
            print("Validation answer:", " ".join(answer_tokens))
        else:
            print("Could not find <sep> token in validation input_ids.")
    else:
        print("No examples in validation set.")
