import torch
import torch.nn as nn
import math  # For math.ceil in parameter estimation if needed

class LLMConfig:
    """Holds the configuration for a toy LLM."""

    def __init__(self, name: str, n_layers: int, hidden_size: int, n_heads: int, total_params_str: str):
        self.name = name
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.total_params_str = total_params_str  # The approximate total parameters as a string (e.g., "~0.4M")

        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")

    def __repr__(self):
        return (f"LLMConfig(name='{self.name}', n_layers={self.n_layers}, "
                f"hidden_size={self.hidden_size}, n_heads={self.n_heads}, "
                f"total_params_str='{self.total_params_str}')")


# List of configurations based on the provided table
llm_configurations_data = [
    {"name": "Tiny", "n_layers": 2, "hidden_size": 128, "n_heads": 2, "total_params_str": "~0.4M"},
    {"name": "Small", "n_layers": 4, "hidden_size": 256, "n_heads": 4, "total_params_str": "~2M"},
    {"name": "Base", "n_layers": 6, "hidden_size": 512, "n_heads": 8, "total_params_str": "~15M"},
    {"name": "Medium", "n_layers": 8, "hidden_size": 768, "n_heads": 12, "total_params_str": "~50M"},
    {"name": "Large", "n_layers": 12, "hidden_size": 1024, "n_heads": 16, "total_params_str": "~100M–150M"},
]

llm_configurations = [LLMConfig(**data) for data in llm_configurations_data]


# --- Conceptual PyTorch-like Model Components ---

class ToyMultiHeadAttention(nn.Module):
    """Conceptual Multi-Head Attention block."""

    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # Dropout can be added here if needed: self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        # This is a placeholder for the actual MHA logic which involves:
        # 1. Projecting Q, K, V
        # 2. Reshaping for multiple heads
        # 3. Scaled dot-product attention: softmax( (Q @ K.transpose) / sqrt(head_dim) ) @ V
        # 4. Concatenating heads
        # 5. Final output projection
        # For simplicity, we'll just pass through the output projection of V

        # Conceptual calculation:
        # B, T, C = q.shape # Batch, Sequence Length, Channels (hidden_size)
        # q_h = self.q_proj(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, T, head_dim)
        # k_h = self.k_proj(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # v_h = self.v_proj(v).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        #
        # att_scores = (q_h @ k_h.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # if mask is not None:
        #     att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
        # att_probs = torch.softmax(att_scores, dim=-1)
        # # att_probs = self.attn_dropout(att_probs) # Apply dropout
        # output = (att_probs @ v_h).transpose(1, 2).contiguous().view(B, T, C)

        # Simplified placeholder for structural definition
        return self.out_proj(self.v_proj(v))  # Not a real MHA computation


class ToyFeedForward(nn.Module):
    """Conceptual Feed-Forward Network (FFN) block."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.activation = nn.GELU()  # GELU is common, ReLU or SiLU are alternatives
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ToyTransformerBlock(nn.Module):
    """Conceptual Transformer Block."""

    def __init__(self, hidden_size: int, n_heads: int, ffn_expansion_factor: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = ToyMultiHeadAttention(hidden_size, n_heads)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after attention

        self.norm2 = nn.LayerNorm(hidden_size)
        ffn_hidden_size = hidden_size * ffn_expansion_factor
        self.ffn = ToyFeedForward(hidden_size, ffn_hidden_size, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after FFN

    def forward(self, x, attention_mask=None):
        # Pre-LayerNorm architecture
        # Attention sub-layer
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=attention_mask)
        x = x + self.dropout1(attn_output)  # Residual connection

        # Feed-forward sub-layer
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)  # Residual connection
        return x


class ToyLLM(nn.Module):
    """Conceptual Toy Language Model."""

    def __init__(self, config: LLMConfig, vocab_size: int, max_seq_len: int,
                 ffn_expansion_factor: int = 4, dropout_rate: float = 0.1, tie_weights: bool = True):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_size)
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            ToyTransformerBlock(config.hidden_size, config.n_heads, ffn_expansion_factor, dropout_rate)
            for _ in range(config.n_layers)
        ])

        # Final LayerNorm and Output Head
        self.norm_out = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, vocab_size, bias=False)  # Bias often False for output head

        # Weight tying (common practice)
        if tie_weights:
            self.output_head.weight = self.token_embedding.weight

        # Initialize weights (important for real models)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # Create positional IDs
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, seq_len)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, hidden_size)
        pos_emb = self.position_embedding(position_ids)  # (1, seq_len, hidden_size)

        x = self.emb_dropout(tok_emb + pos_emb)  # Add token and position embeddings

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)  # Assuming attention_mask is correctly shaped if provided

        x = self.norm_out(x)
        logits = self.output_head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    def estimate_parameters(self, ffn_expansion_factor: int = 4, tie_weights: bool = True) -> int:
        """
        Estimates the number of parameters in the model based on its structure.
        This is a formula-based estimation for the conceptual model.
        For an actual PyTorch model, sum(p.numel() for p in model.parameters() if p.requires_grad) is used.
        """
        h = self.config.hidden_size
        num_layers = self.config.n_layers

        params = 0

        # 1. Token Embeddings
        params += self.vocab_size * h

        # 2. Positional Embeddings
        params += self.max_seq_len * h

        # 3. Transformer Blocks
        # Each block contains:
        #   - MHA: 4 * (h*h for weights + h for biases) = 4*h^2 + 4*h
        #   - LayerNorm1: 2*h (weight + bias)
        #   - FFN:
        #       - linear1: h * (h*ffn_expansion_factor) + (h*ffn_expansion_factor) for bias
        #       - linear2: (h*ffn_expansion_factor) * h + h for bias
        #       Total FFN: 2 * h^2 * ffn_expansion_factor + h * ffn_expansion_factor + h
        #   - LayerNorm2: 2*h (weight + bias)

        params_per_block = 0
        # MHA (q,k,v,out projections, each with weight and bias)
        params_per_block += 4 * (h * h + h)
        # FFN
        ffn_h = h * ffn_expansion_factor
        params_per_block += (h * ffn_h + ffn_h)  # linear1 + bias1
        params_per_block += (ffn_h * h + h)  # linear2 + bias2
        # LayerNorms (2 per block, each with weight and bias)
        params_per_block += 2 * (2 * h)

        params += num_layers * params_per_block

        # 4. Final LayerNorm
        params += 2 * h

        # 5. Output Head (if weights are not tied)
        if not tie_weights:
            # output_head is nn.Linear(config.hidden_size, vocab_size, bias=False)
            params += h * self.vocab_size
            # if bias was True: params += self.vocab_size

        return params


# --- Example Usage ---
if __name__ == "__main__":
    print("Defined LLM Configurations:")
    for config in llm_configurations:
        print(config)
    print("-" * 30)

    # Example: Instantiate the "Tiny" model
    tiny_config = next(c for c in llm_configurations if c.name == "Tiny")

    # Define typical vocabulary size and max sequence length for estimation
    # These values significantly impact total parameters.
    # The "~0.4M" for Tiny might assume a very small vocab or exclude embeddings.
    example_vocab_size = 32000  # A common vocab size
    example_max_seq_len = 512  # Max tokens the model can process

    # Try different FFN expansion factors to see impact on params
    # Common values are 4, but smaller models sometimes use 2 or 8/3.
    # The original GPT-2 used ffn_expansion_factor = 4
    # The values in the table might imply a specific ffn_expansion_factor or other architectural choices.

    print(f"\nInstantiating '{tiny_config.name}' model with:")
    print(f"  Vocab Size: {example_vocab_size}")
    print(f"  Max Sequence Length: {example_max_seq_len}")

    ffn_exp_factors_to_test = [2, 4,
                               8 / 3]  # 8/3 is approx 2.66, often used in Llama-like models as (2/3 * 4) with SwiGLU

    for ffn_factor in ffn_exp_factors_to_test:
        actual_ffn_factor = math.ceil(ffn_factor) if isinstance(ffn_factor,
                                                                float) else ffn_factor  # Ensure integer for layer dim
        # For SwiGLU type FFNs, the actual hidden dim is often 2/3 of ffn_expansion_factor * hidden_size,
        # but involves 3 matrices instead of 2.
        # Here, ffn_expansion_factor directly multiplies hidden_size for the intermediate FFN layer.

        tiny_llm_model = ToyLLM(
            config=tiny_config,
            vocab_size=example_vocab_size,
            max_seq_len=example_max_seq_len,
            ffn_expansion_factor=actual_ffn_factor,  # Standard FFN expansion
            tie_weights=True
        )

        estimated_params = tiny_llm_model.estimate_parameters(
            ffn_expansion_factor=actual_ffn_factor,
            tie_weights=True
        )
        print(f"\n  With FFN Expansion Factor: {ffn_factor:.2f} (actual intermediate multiplier: {actual_ffn_factor})")
        print(
            f"  Estimated Parameters for '{tiny_config.name}': {estimated_params:,} (approx {estimated_params / 1e6:.2f}M)")
        print(f"  Table's 'Total Params': {tiny_config.total_params_str}")

    print("-" * 30)
    # To get parameters from an actual PyTorch model instance:
    # total_params_pytorch = sum(p.numel() for p in tiny_llm_model.parameters() if p.requires_grad)
    # print(f"  PyTorch calculated params for last instance: {total_params_pytorch:,}")

    # Example for "Small" model to try and match ~2M (often implies core params or small vocab)
    small_config = next(c for c in llm_configurations if c.name == "Small")
    # Let's assume a much smaller vocab for "toy" context matching
    toy_vocab_size = 1000
    toy_max_seq_len = 256

    print(f"\nInstantiating '{small_config.name}' model with (toy settings):")
    print(f"  Vocab Size: {toy_vocab_size}")
    print(f"  Max Sequence Length: {toy_max_seq_len}")

    for ffn_factor in [2, 4]:  # Test FFN expansion 2 and 4
        small_llm_model_toy = ToyLLM(
            config=small_config,
            vocab_size=toy_vocab_size,
            max_seq_len=toy_max_seq_len,
            ffn_expansion_factor=ffn_factor,
            tie_weights=True
        )
        estimated_params_small_toy = small_llm_model_toy.estimate_parameters(
            ffn_expansion_factor=ffn_factor,
            tie_weights=True
        )
        print(f"\n  With FFN Expansion Factor: {ffn_factor}")
        print(
            f"  Estimated Parameters for '{small_config.name}' (toy settings): {estimated_params_small_toy:,} (approx {estimated_params_small_toy / 1e6:.2f}M)")
        print(f"  Table's 'Total Params': {small_config.total_params_str}")
        # If we only count core transformer block parameters (n_layers * (12 * h^2) for ffn_exp=4 or n_layers * (8 * h^2) for ffn_exp=2)
        # this value would be much closer to the table.
        # Small (core, ffn_exp=2): 4 layers * (8 * 256^2 + biases) ~ 2.1M
        # Small (core, ffn_exp=4): 4 layers * (12 * 256^2 + biases) ~ 3.1M
        # The table's ~2M for Small suggests an FFN expansion closer to 2 for core block parameters,
        # or that the "Total Params" heavily focuses on block parameters.
