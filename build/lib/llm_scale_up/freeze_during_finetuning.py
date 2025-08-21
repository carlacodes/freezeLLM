import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- Configuration Class and Toy LLM Definitions (from previous code) ---


class LLMConfig:
    """Holds the configuration for a toy LLM."""

    def __init__(
        self,
        name: str,
        n_layers: int,
        hidden_size: int,
        n_heads: int,
        total_params_str: str,
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
            f"hidden_size={self.hidden_size}, n_heads={self.n_heads}, "
            f"total_params_str='{self.total_params_str}')"
        )


class ToyMultiHeadAttention(nn.Module):
    """Conceptual Multi-Head Attention block."""

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

    def forward(self, q, k, v, mask=None):  # Placeholder
        return self.out_proj(self.v_proj(v))


class ToyFeedForward(nn.Module):
    """Conceptual Feed-Forward Network (FFN) block."""

    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # Placeholder
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class ToyTransformerBlock(nn.Module):
    """Conceptual Transformer Block."""

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

    def forward(self, x, attention_mask=None):  # Placeholder
        attn_output = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask=attention_mask
        )
        x = x + self.dropout1(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        return x


class ToyLLM(nn.Module):
    """Conceptual Toy Language Model."""

    def __init__(
        self,
        config: LLMConfig,
        vocab_size: int,
        max_seq_len: int,
        ffn_expansion_factor: int = 4,
        dropout_rate: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, config.hidden_size)
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
        self.output_head = nn.Linear(config.hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.output_head.weight = self.token_embedding.weight
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

    def forward(self, input_ids, attention_mask=None):  # Placeholder
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        x = self.emb_dropout(
            self.token_embedding(input_ids) + self.position_embedding(position_ids)
        )
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.output_head(self.norm_out(x))


def track_weight_dynamics(
    model: ToyLLM, num_epochs: int, perturbation_strength: float = 0.001
):
    """
    Simulates fine-tuning and tracks weight dynamics over epochs.

    Args:
        model: The PyTorch model (ToyLLM instance) to analyze.
        num_epochs: The number of "fine-tuning" epochs to simulate.
        perturbation_strength: A small factor to control the magnitude of random
                               changes applied to weights each epoch.

    Returns:
        A tuple containing:
        - weight_variations (dict): {param_name: [L2_norm_delta_epoch_1, ...], ...}
                                   Tracks L2 norm of (W_curr - W_prev).
        - weight_magnitudes (dict): {param_name: [L2_norm_epoch_0, L2_norm_epoch_1, ...], ...}
                                   Tracks L2 norm of W at each epoch (including initial).
    """
    weight_variations = {name: [] for name, _ in model.named_parameters()}
    weight_magnitudes = {name: [] for name, _ in model.named_parameters()}

    # Store initial state of weights (deep copy for subsequent comparisons)
    # This represents weights at epoch 0, before any "fine-tuning"
    previous_state_dict = copy.deepcopy(model.state_dict())

    # Record initial L2 magnitudes (epoch 0)
    for name, param in model.named_parameters():
        weight_magnitudes[name].append(torch.norm(param.data, p=2).item())

    print(f"Starting conceptual fine-tuning for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # --- 1. Simulate a weight update for the current epoch ---
        # In real fine-tuning, an optimizer updates weights based on gradients from data.
        # Here, we perturb weights slightly to simulate change.
        with torch.no_grad():
            for (
                name_param,
                param,
            ) in (
                model.named_parameters()
            ):  # Use named_parameters to potentially vary strength
                # Additive random noise scaled by perturbation_strength
                # You could make perturbation_strength adaptive or different per layer type if desired
                change = torch.randn_like(param.data) * perturbation_strength
                param.data += change

        # --- 2. Calculate and store variation and current magnitudes ---
        current_state_dict = model.state_dict()  # Get weights after simulated update
        for name, param_current_tensor in current_state_dict.items():
            # Ensure we only process parameters that were in previous_state_dict
            if name not in previous_state_dict:
                continue

            param_previous_tensor = previous_state_dict[name]

            # Calculate L2 norm of the difference (variation)
            # This is || W_current_epoch - W_previous_epoch ||_2
            diff = (
                param_current_tensor.data - param_previous_tensor.data
            )  # Use .data to avoid graph issues
            variation = torch.norm(diff, p=2).item()
            if name in weight_variations:  # Make sure the key exists (it should)
                weight_variations[name].append(variation)

            # Calculate and store L2 norm of current weights (magnitude)
            current_magnitude = torch.norm(param_current_tensor.data, p=2).item()
            if name in weight_magnitudes:
                weight_magnitudes[name].append(current_magnitude)

        # Update previous_state_dict for the next epoch's comparison
        previous_state_dict = copy.deepcopy(current_state_dict)

        # Optional: Print some summary for the epoch
        total_variation_this_epoch = sum(
            variations[-1] for variations in weight_variations.values() if variations
        )
        print(
            f"Epoch {epoch}/{num_epochs}: Sum of L2 weight deltas = {total_variation_this_epoch:.4e}"
        )

    return weight_variations, weight_magnitudes


def plot_weight_dynamics_matplotlib(
    data_dict: dict,
    title_prefix: str,
    num_epochs: int,
    is_variation_data: bool,
    params_to_plot: list = None,
    log_scale: bool = False,
):
    """
    Plots the collected weight dynamics (variations or magnitudes) using matplotlib.

    Args:
        data_dict (dict): Dictionary containing the data series for each parameter.
                          e.g., {param_name: [value_epoch_1, ..., value_epoch_N]}
        title_prefix (str): Prefix for the plot titles (e.g., "Weight Variation" or "Weight Magnitude").
        num_epochs (int): Total number of epochs simulated.
        is_variation_data (bool): True if data is variation (num_epochs points),
                                 False if data is magnitude (num_epochs+1 points including initial).
        params_to_plot (list, optional): Specific parameter names to plot. If None, selects a few.
        log_scale (bool, optional): Whether to use a logarithmic scale for the y-axis.
    """
    if not plt:
        print("Matplotlib is not available. Skipping plotting.")
        return

    if is_variation_data:
        epochs_range = range(
            1, num_epochs + 1
        )  # Variations are calculated from epoch 1 onwards
    else:
        epochs_range = range(
            num_epochs + 1
        )  # Magnitudes include initial state at epoch 0

    if params_to_plot is None:
        all_param_names = list(data_dict.keys())
        # Select a few diverse parameters if none are specified (e.g., embedding, first/mid/last layer components)
        if len(all_param_names) > 4:
            indices_to_plot = [
                0,
                len(all_param_names) // 4,
                len(all_param_names) // 2,
                min(
                    len(all_param_names) - 1, (3 * len(all_param_names)) // 4
                ),  # ensure it's a valid index
                len(all_param_names) - 1,
            ]
            # Remove duplicates if list is small
            indices_to_plot = sorted(list(set(indices_to_plot)))

            params_to_plot = [
                all_param_names[i] for i in indices_to_plot if i < len(all_param_names)
            ]
            params_to_plot = [
                p for p in params_to_plot if p in data_dict and data_dict[p]
            ]  # Ensure data exists
        else:
            params_to_plot = [
                p for p in all_param_names if p in data_dict and data_dict[p]
            ]

    if not params_to_plot:
        print(f"No parameters selected or available for plotting for {title_prefix}.")
        return

    num_selected_plots = len(params_to_plot)
    fig, axes = plt.subplots(
        num_selected_plots, 1, figsize=(14, num_selected_plots * 4), sharex=True
    )
    if num_selected_plots == 1:  # Ensure axes is always iterable
        axes = [axes]

    for i, param_name in enumerate(params_to_plot):
        if param_name not in data_dict or not data_dict[param_name]:
            print(f"No data for {param_name}, skipping plot.")
            if (
                num_selected_plots == 1
            ):  # If only one subplot was planned and it's skipped.
                fig.clf()  # Clear the figure to avoid empty plot
                plt.close(fig)
                print(f"No data to plot for {title_prefix}")
                return
            continue

        ax = axes[i]
        data_series = data_dict[param_name]

        # Ensure data_series length matches expected epoch range
        if is_variation_data and len(data_series) != num_epochs:
            print(
                f"Warning: Mismatch length for variation data {param_name}. "
                f"Expected {num_epochs}, got {len(data_series)}"
            )
            # continue # or try to plot what's available
        elif not is_variation_data and len(data_series) != num_epochs + 1:
            print(
                f"Warning: Mismatch length for magnitude data {param_name}. "
                f"Expected {num_epochs + 1}, got {len(data_series)}"
            )
            # continue

        ax.plot(
            epochs_range[: len(data_series)], data_series, marker="o", linestyle="-"
        )  # Plot only available data points
        ax.set_ylabel(f"{title_prefix}\n({param_name})", fontsize=9)
        # ax.set_title(f"{title_prefix} for {param_name}", fontsize=10) # Title can be redundant with suptitle
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.5)

    if num_selected_plots > 0 and any(
        ax.has_data() for ax in axes
    ):  # Check if any axes actually have data
        axes[-1].set_xlabel("Epoch")
        fig.suptitle(
            f"{title_prefix} Dynamics During Conceptual Fine-tuning", fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout
        plt.show()
    else:
        print(f"No data plotted for {title_prefix}")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define LLM configurations (example)
    llm_configurations_data = [
        {
            "name": "Tiny",
            "n_layers": 2,
            "hidden_size": 128,
            "n_heads": 2,
            "total_params_str": "~0.4M",
        },
        {
            "name": "Small",
            "n_layers": 4,
            "hidden_size": 256,
            "n_heads": 4,
            "total_params_str": "~2M",
        },
    ]
    llm_configs = [LLMConfig(**data) for data in llm_configurations_data]

    # 2. Select a model configuration and instantiate the model
    # Let's use the "Tiny" model for a quick example
    chosen_config = llm_configs[0]
    example_vocab_size = 1000  # Smaller vocab for toy example
    example_max_seq_len = 64  # Shorter sequence length

    toy_model = ToyLLM(
        config=chosen_config,
        vocab_size=example_vocab_size,
        max_seq_len=example_max_seq_len,
    )
    print(f"Instantiated Model: {chosen_config.name}")
    num_params = sum(p.numel() for p in toy_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # 3. Set parameters for the simulation
    NUM_SIMULATED_EPOCHS = 10
    # Adjust this factor; too small and changes are negligible, too large and it's just noise.
    # It represents how much weights change per epoch on average.
    WEIGHT_PERTURBATION_STRENGTH = (
        0.0005  # e.g., 0.05% random change relative to a N(0,1) draw
    )

    # 4. Run the tracking function
    variations, magnitudes = track_weight_dynamics(
        toy_model,
        NUM_SIMULATED_EPOCHS,
        perturbation_strength=WEIGHT_PERTURBATION_STRENGTH,
    )

    # 5. Plot the results (make sure you have matplotlib installed: pip install matplotlib)
    # Plot L2 norm of weight deltas (variations)
    plot_weight_dynamics_matplotlib(
        variations,
        title_prefix="L2 Norm of Weight Delta (Variation)",
        num_epochs=NUM_SIMULATED_EPOCHS,
        is_variation_data=True,
        log_scale=True,  # Variations can span orders of magnitude, log scale is often useful
    )

    # Plot L2 norm of actual weights (magnitudes)
    plot_weight_dynamics_matplotlib(
        magnitudes,
        title_prefix="L2 Norm of Weights (Magnitude)",
        num_epochs=NUM_SIMULATED_EPOCHS,
        is_variation_data=False,
        log_scale=False,
    )

    print("\nAnalysis complete. Check the plots for weight dynamics.")
    print("Note: The 'fine-tuning' is simulated by adding small random perturbations.")
    print(
        "In a real scenario, weight changes would be driven by data and a loss function."
    )
