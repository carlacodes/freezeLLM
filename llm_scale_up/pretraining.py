import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator


class LLMConfig:
    """Holds the configuration for a toy LLM."""

    def __init__(
        self,
        name: str,
        n_layers: int,
        hidden_size: int,
        n_heads: int,
        total_params_str: str = "",
    ):  # Added default for total_params_str
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
        )  # Removed total_params_str for brevity


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

    def forward(self, q, k, v, attention_mask=None):  # Added attention_mask
        B, T, C = q.shape
        q_h = self.q_proj(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(v).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att_scores = (q_h @ k_h.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            # Attention mask should be (B, 1, T, T) or (B, n_heads, T, T)
            # And have 0s for positions to attend to, -inf for positions to mask.
            # Or True for positions to keep, False for positions to mask.
            # Assuming mask is (B, T) -> (B, 1, 1, T) for key padding mask
            # Or (B, T, T) -> (B, 1, T, T) for combined mask
            # For MLM, usually an attention_mask is for padding, not causal.
            # If attention_mask is (B, T) where 0 is padding:
            # mask shape (B, 1, 1, T) for scores (B, H, T_q, T_k)
            if attention_mask.dim() == 2:  # (B, T_k)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                    2
                )  # (B, 1, 1, T_k)
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

    def forward(self, x, attention_mask=None):  # Pass attention_mask
        # Pre-LayerNorm
        normed_x = self.norm1(x)
        attn_output = self.attention(
            normed_x, normed_x, normed_x, attention_mask=attention_mask
        )
        x = x + self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        return x


class ToyLLM(nn.Module):
    """Conceptual Toy Language Model, now ready for pre-training."""

    def __init__(
        self,
        config: LLMConfig,
        vocab_size: int,
        max_seq_len: int,
        pad_token_id: int,
        ffn_expansion_factor: int = 4,
        dropout_rate: float = 0.1,
    ):  # Added pad_token_id
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id  # Store pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, config.hidden_size, padding_idx=pad_token_id
        )  # Use padding_idx
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
        # For MLM, this head predicts the original token for masked positions
        self.mlm_output_head = nn.Linear(
            config.hidden_size, vocab_size
        )  # Renamed for clarity

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Don't re-initialize padding_idx row
            padding_idx = module.padding_idx
            temp_weight = torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if padding_idx is not None:
                with torch.no_grad():
                    temp_weight[padding_idx].fill_(0)
            module.weight.data = temp_weight

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len), 1 for non-padded, 0 for padded
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

        # Create attention mask for self-attention if not provided based on padding
        # The TransformerBlock's MHA expects mask where 0 means 'mask this key out'
        # input `attention_mask` is usually 1 for token, 0 for pad.
        # MHA needs mask: (B, T, T) or (B, H, T, T).
        # For padding: (B, 1, 1, T_key) where masked positions are -inf (or 0 for boolean)
        # Our input attention_mask is (B, T_key) where 0 = pad.
        # This is handled inside ToyMultiHeadAttention.

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)  # Pass the padding mask

        x = self.norm_out(x)
        logits = self.mlm_output_head(x)  # (batch_size, seq_len, vocab_size)
        return logits


# --- Data Loading and Preprocessing for WikiText-2 with MLM ---
TOKENIZER = get_tokenizer("basic_english")
UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN = (
    "<unk>",
    "<pad>",
    "<mask>",
    "<bos>",
    "<eos>",
)
SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def yield_tokens(data_iter, tokenizer_fn):
    for text in data_iter:
        yield tokenizer_fn(text)


def build_vocab_from_wikitext2(min_freq=5):
    train_iter, _, _ = WikiText2()  # Using train split to build vocab
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, TOKENIZER), min_freq=min_freq, specials=SPECIAL_TOKENS
    )
    vocab.set_default_index(vocab[UNK_TOKEN])
    return vocab


def create_mlm_inputs_and_labels(token_ids, vocab, mlm_probability=0.15):
    """
    Prepares inputs and labels for MLM.
    token_ids: tensor of shape (seq_len) or (batch_size, seq_len)
    """
    mask_token_id = vocab[MASK_TOKEN]
    pad_token_id = vocab[PAD_TOKEN]

    inputs = token_ids.clone()
    labels = token_ids.clone()

    # Probability of masking
    prob_matrix = torch.full(labels.shape, mlm_probability)

    # Don't mask special tokens like PAD
    special_tokens_mask = (
        (labels == pad_token_id)
        | (labels == vocab[BOS_TOKEN])
        | (labels == vocab[EOS_TOKEN])
        | (labels == mask_token_id)
    )  # Don't mask existing masks

    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()

    # For 80% of masked positions, replace with MASK token
    inputs[masked_indices] = mask_token_id

    # For 10% of masked positions, replace with a random token
    # (This part is more complex to implement efficiently in batch without loops,
    # for simplicity in a toy example, we might simplify or stick to 80% MASK, 20% same)
    # Let's do 80% MASK, 10% random, 10% same
    indices_replaced = masked_indices & (
        torch.rand(labels.shape, device=labels.device) < 0.8
    )
    indices_random = (
        masked_indices
        & ~indices_replaced
        & (torch.rand(labels.shape, device=labels.device) < 0.5)
    )  # 0.5 of remaining 20% = 10% total
    # indices_same = masked_indices & ~indices_replaced & ~indices_random (implicitly the rest 10%)

    inputs[indices_replaced] = mask_token_id
    random_words = torch.randint(
        len(vocab), labels.shape, dtype=torch.long, device=labels.device
    )
    inputs[indices_random] = random_words[indices_random]
    labels[~masked_indices] = -100  # CrossEntropyLoss default ignore_index

    return inputs, labels


def collate_batch_mlm(batch, vocab, tokenizer_fn, max_seq_len, device):
    bos_token_id = vocab[BOS_TOKEN]
    eos_token_id = vocab[EOS_TOKEN]
    pad_token_id = vocab[PAD_TOKEN]

    processed_texts = []
    for text_sample in batch:  # batch is a list of raw text strings
        if not text_sample.strip():
            continue  # Skip empty lines
        tokens = (
            [bos_token_id]
            + [vocab[token] for token in tokenizer_fn(text_sample)]
            + [eos_token_id]
        )
        processed_texts.append(torch.tensor(tokens, dtype=torch.long))

    if not processed_texts:  # Handle empty batch after filtering
        return None, None, None

    # Pad sequences to max_seq_len for this batch or global max_seq_len
    # Truncate if longer than max_seq_len
    padded_sequences = [p[:max_seq_len] for p in processed_texts]
    padded_sequences = pad_sequence(
        padded_sequences, batch_first=True, padding_value=pad_token_id
    )

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_masks = (padded_sequences != pad_token_id).long()

    # Create MLM inputs and labels
    mlm_inputs, mlm_labels = create_mlm_inputs_and_labels(padded_sequences, vocab)

    return mlm_inputs.to(device), mlm_labels.to(device), attention_masks.to(device)


# --- Pre-training Loop ---
def pretrain_wikitext2_epoch(
    model, dataloader, optimizer, criterion, vocab, device, epoch_num, log_interval=100
):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (mlm_inputs, mlm_labels, attention_masks) in enumerate(dataloader):
        if mlm_inputs is None:
            continue  # Skip if collate_fn returned empty

        optimizer.zero_grad()
        output_logits = model(
            mlm_inputs, attention_mask=attention_masks
        )  # (batch, seq_len, vocab_size)

        # Calculate loss only for masked tokens
        # Reshape for CrossEntropyLoss: (N, C) where C = number of classes
        # output_logits.view(-1, vocab_size)
        # mlm_labels.view(-1)
        loss = criterion(output_logits.view(-1, model.vocab_size), mlm_labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            print(
                f"Epoch {epoch_num} | Batch {batch_idx}/{len(dataloader)} | Avg Loss: {avg_loss:.4f}"
            )
            # Reset for next logging interval within epoch if desired
            # total_loss = 0
            # num_batches = 0

    return total_loss / num_batches if num_batches > 0 else 0


# --- Example Usage: Pre-training ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Build Vocabulary from WikiText-2
    print("Building vocabulary from WikiText-2...")
    vocab = build_vocab_from_wikitext2(min_freq=5)
    VOCAB_SIZE = len(vocab)
    PAD_TOKEN_ID = vocab[PAD_TOKEN]
    MASK_TOKEN_ID = vocab[
        MASK_TOKEN
    ]  # For MLM input creation (though create_mlm_inputs_and_labels uses vocab[MASK_TOKEN])
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"PAD token ID: {PAD_TOKEN_ID}, MASK token ID: {MASK_TOKEN_ID}")

    # 2. Model Configuration and Instantiation
    # Using a "Tiny" configuration for quick demonstration
    tiny_config = LLMConfig(name="TinyPretrain", n_layers=2, hidden_size=128, n_heads=2)
    MAX_SEQ_LEN = 128  # Max sequence length for pre-training and positional embeddings
    DROPOUT_RATE = 0.1

    toy_llm_pretrained = ToyLLM(
        config=tiny_config,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        pad_token_id=PAD_TOKEN_ID,  # Pass pad_token_id
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    num_params = sum(
        p.numel() for p in toy_llm_pretrained.parameters() if p.requires_grad
    )
    print(
        f"Instantiated Model: {tiny_config.name} with {num_params:,} trainable parameters."
    )

    # For this example, let's process the training data into a list of strings.
    train_iter_raw, valid_iter_raw, test_iter_raw = WikiText2()
    train_data_list = [
        line for line in train_iter_raw if line.strip()
    ]  # Collect non-empty lines
    # You would typically use valid_iter for validation during pre-training.

    BATCH_SIZE = 16  # Adjust based on your GPU memory

    collate_fn_partial = lambda batch: collate_batch_mlm(
        batch, vocab, TOKENIZER, MAX_SEQ_LEN, DEVICE
    )

    train_dataloader = DataLoader(
        train_data_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_partial,
    )
    valid_dataloader = DataLoader(
        list(valid_iter_raw),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_partial,
    )

    LEARNING_RATE = 1e-4
    optimizer = optim.AdamW(toy_llm_pretrained.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    NUM_PRETRAIN_EPOCHS = 1000
    print(f"\nStarting MLM pre-training for {NUM_PRETRAIN_EPOCHS} epochs...")

    for epoch in range(1, NUM_PRETRAIN_EPOCHS + 1):
        # --- Training Step ---
        # (Ensure toy_llm_pretrained is on DEVICE before starting training if not done already)
        # toy_llm_pretrained.to(DEVICE)
        # The pretrain_wikitext2_epoch function should internally call toy_llm_pretrained.train()
        avg_epoch_loss = pretrain_wikitext2_epoch(
            toy_llm_pretrained,  # Your actual model instance
            train_dataloader,
            optimizer,
            criterion,
            vocab,
            DEVICE,
            epoch_num=epoch,
            log_interval=500,
        )
        print(
            f"--- End of Epoch {epoch} | Average Pre-training Loss: {avg_epoch_loss:.4f} ---"
        )

        # --- Validation Step ---
        toy_llm_pretrained.eval()  # Set YOUR model to evaluation mode
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():  # No gradients needed for validation
            for (
                mlm_inputs,
                mlm_labels,
                attention_masks,
            ) in valid_dataloader:  # Ensure valid_dataloader is defined
                if mlm_inputs is None:
                    continue  # If collate_fn can return None for empty batches

                # Assuming mlm_inputs, mlm_labels, attention_masks are already on DEVICE
                # from your collate_fn_partial
                output_logits = toy_llm_pretrained(
                    mlm_inputs, attention_mask=attention_masks
                )
                loss = criterion(
                    output_logits.view(-1, toy_llm_pretrained.vocab_size),
                    mlm_labels.view(-1),
                )
                total_val_loss += loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            print(
                f"--- Epoch {epoch} | Average Validation Loss: {avg_val_loss:.4f} ---"
            )
        else:
            print(
                f"--- Epoch {epoch} | No data in validation loader or all batches skipped. ---"
            )

        toy_llm_pretrained.train()  # Set YOUR model back to training mode for the next epoch

    print("\nPre-training finished.")

    PRETRAINED_MODEL_PATH = "toy_llm_wikitext2_pretrained.pth"
    torch.save(toy_llm_pretrained.state_dict(), PRETRAINED_MODEL_PATH)
    print(f"Pre-trained model state_dict saved to {PRETRAINED_MODEL_PATH}")

    # --- Next Steps ---
    # This `toy_llm_pretrained` model now has weights learned from WikiText-2.
    # You would then:
    # 1. Load this model's state_dict for fine-tuning on `/luheng/qa_srl`.
    # 2. Implement a fine-tuning loop specific to your QA-SRL task
    # (new dataset, new head for the model if needed, new loss).
    # 3. You could use the `track_weight_dynamics` function (from previous discussions)
    #    during that fine-tuning phase to observe how these pre-trained weights change.
