# Emergence of Semantic Role Understanding in LLMs

## Abstract

Large language models trained on next-token prediction often exhibit capabilities beyond what they were explicitly trained for. This study investigates whether semantic role understanding—the ability to answer questions like "Who sat on the log?" or "What did someone give to whom?"—emerges from pretraining on semantically rich text. Using a linear probing methodology, we freeze pretrained LLM weights and train only a minimal QA head to test whether predicate-argument structure is already encoded in the representations. We compare performance across model scales (500K to 100M+ parameters) to identify emergence thresholds, and ablate pretraining data to isolate the contribution of general text versus QA-formatted examples. If frozen models perform comparably to fully finetuned models, this provides evidence that logical reasoning about semantic roles emerges from language modeling alone, rather than requiring explicit supervision.

---

## Quick Start: Exact Modal Commands

**Prerequisites:**
1. Install Modal: `pip install modal`
2. Authenticate: `modal token new`

**Navigate to script directory:**
```bash
cd llm_scale_up
```

### Minimum Viable Experiment (Start Here)

Run these commands to test emergence on the `tiny` model:

```bash
# 1. Baseline: Can tiny model learn the task at all? (runs pretrain + finetune)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny
# Note the output folder, e.g. TinyQA_spe200k_20251231-005835/

# 2. Linear probe: Does emergence occur? (uses pretrained model from step 1)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm --skip-pretrain \
  --pretrained-model-path "TinyQA_spe200k_20251231-005835/toy_llm_unified_pretrained.pth"

# 3. Random baseline: Establish performance floor (skips pretraining entirely)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --random-baseline --skip-pretrain
```

### Full Experiment (All Model Sizes)

```bash
# Phase 1: Baseline capability check (full finetuning)
# Note the output folder for each (e.g., TinyQA_spe200k_YYYYMMDD-HHMMSS/)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium

# Phase 2: Test for emergence (linear probe / freeze LLM)
# Use --skip-pretrain and --pretrained-model-path to reuse pretrained models from Phase 1
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm --skip-pretrain \
  --pretrained-model-path "TinyQA_spe200k_YYYYMMDD-HHMMSS/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small --freeze-llm --skip-pretrain \
  --pretrained-model-path "SmallQA_spe400k_YYYYMMDD-HHMMSS/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm --skip-pretrain \
  --pretrained-model-path "BaseQA_spe800k_YYYYMMDD-HHMMSS/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium --freeze-llm --skip-pretrain \
  --pretrained-model-path "MediumQA_spe1500k_YYYYMMDD-HHMMSS/toy_llm_unified_pretrained.pth"
```

### Listing Saved Models

```bash
modal run pretrain_wikitext_finetune_qasrl_modal.py --list-saved
```

---

## Post-Training Analysis Steps

After running `modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size>`, follow these steps to collect results and run analyses.

### Step 1: Verify Training Completed Successfully

```bash
# List all saved models and check for your experiment
modal run pretrain_wikitext_finetune_qasrl_modal.py --list-saved
```

Look for directories like:
- `TinyQA_spe200k_20251230-120000/` (full finetuning)
- `TinyQA_spe200k_frozen_20251230-130000/` (frozen/linear probe)
- `TinyQA_spe200k_random_20251230-140000/` (random baseline)

Each directory should contain:
- `toy_llm_unified_pretrained.pth` - Pretrained model weights
- `toy_llm_qasrl_finetuned.pth` - Best finetuned model weights
- `finetune_stats.json` - Final metrics and filtering statistics
- `finetune_epoch_*.pth` - Epoch snapshots for CKA analysis
- `pretrain_epoch_*.pth` - Pretraining snapshots (if applicable)

### Step 2: Collect Results from finetune_stats.json

Download and inspect the results:

```bash
# Download stats file for a specific model
modal volume get llm-models <ModelDir>/finetune_stats.json ./results/

# Example for tiny model
modal volume get llm-models TinyQA_spe200k_20251230-120000/finetune_stats.json ./results/tiny_full.json
modal volume get llm-models TinyQA_spe200k_frozen_20251230-130000/finetune_stats.json ./results/tiny_frozen.json
```

The `finetune_stats.json` contains:
```json
{
  "config_name": "tiny",
  "finetune_epochs": 15,
  "best_validation_loss": 2.34,
  "final_validation_accuracy": 0.42,
  "final_validation_f1": 0.51,
  "freeze_llm": false,
  "random_baseline": false,
  "finetune_samples_limit": null,
  "qasrl_train_filtering": {
    "total_raw_examples": 6414,
    "valid_examples": 5823,
    "skipped_ambiguous_span": 312
  }
}
```

### Step 3: Run Random Baseline (Required for Emergence Score)

The random baseline establishes the performance floor (no pretraining, random init):

```bash
# Run random baseline for each model size (skips pretraining entirely)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --random-baseline --skip-pretrain
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small --random-baseline --skip-pretrain
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --random-baseline --skip-pretrain
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium --random-baseline --skip-pretrain
```

### Step 4: Compute Emergence Metrics

Using results from Steps 2-3, calculate:

```python
# Normalized Emergence Score (from paper Section 3.4)
def compute_emergence_score(f1_frozen, f1_full, f1_random):
    """
    Returns value between 0 and 1.
    - Near 1.0: Strong emergence (frozen captures most capability)
    - Near 0.0: No emergence (frozen no better than random)
    """
    return (f1_frozen - f1_random) / (f1_full - f1_random)

# Raw Emergence Gap
def compute_emergence_gap(f1_full, f1_frozen):
    """Smaller gap = more emergence."""
    return f1_full - f1_frozen
```

### Step 5: Run Temporal CKA Analysis

Analyze how representations change during finetuning:

```bash
# Run CKA analysis for a trained model
modal run temporal_cka_analysis.py --model-dir TinyQA_spe200k_20251230-120000 --config-name tiny

# Compare frozen vs full finetuning
modal run temporal_cka_analysis.py --model-dir TinyQA_spe200k_frozen_20251230-130000 --config-name tiny
```

This generates:
- Layer-wise CKA heatmaps (how each layer changes during training)
- Null model comparison (drift from pretrained baseline)
- Random baseline comparison (structure vs random weights)

### Step 6: Verify QA-SRL Data Alignment (Optional)

Run the verification script to confirm answer span alignment:

```bash
python verify_qasrl_alignment.py
```

This validates that token positions correctly map to expected answer text.

### Step 7: Compile Results Table

Create a results table for all experiments:

| Model | Pretrain | Mode | F1 | EM | Emergence Score |
|-------|----------|------|-----|-----|-----------------|
| tiny | wiki | full | X.XX | X.XX | - |
| tiny | wiki | frozen | X.XX | X.XX | X.XX |
| tiny | wiki | random | X.XX | X.XX | - |
| small | wiki | full | X.XX | X.XX | - |
| small | wiki | frozen | X.XX | X.XX | X.XX |
| ... | ... | ... | ... | ... | ... |

### Step 8: Generate Paper Figures

```bash
# Download all epoch snapshots for learning curve plots
modal volume get llm-models TinyQA_spe200k_20251231-005835/ ./checkpoints/tiny_full/

# Plot training curves, CKA heatmaps, emergence vs scale
python scripts/generate_figures.py --results-dir ./results/
```

---

## Research Question

**Can LLMs develop the ability to answer "who-did-what-to-whom" questions (semantic role understanding) from pretraining on semantically rich text alone, without explicit QA training?**

This investigates whether logical reasoning about predicate-argument structure **emerges** from language modeling, rather than being explicitly learned during finetuning.

---

## Key Concepts

### Emergence vs Transfer Learning

| Concept | Definition | How to Detect |
|---------|------------|---------------|
| **Emergence** | Capability appears in pretrained representations without explicit training | Frozen LLM + simple probe performs well |
| **Transfer** | Capability is learned during finetuning | Full finetuning required for good performance |

### Linear Probe (Freeze LLM)

When `--freeze-llm` is enabled:
- All pretrained LLM weights are frozen (no gradient updates)
- Only the QA head (a single linear layer) is trained
- Tests: "Does the pretrained model already encode answer locations?"

If the linear probe works well, the semantic role information **emerged** during pretraining.

---

## Experimental Design

### Phase 1: Baseline Capability Check

**Goal:** Determine which model sizes can learn the task at all.

```bash
# Run for each model size with full finetuning
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny
modal run -d pretrain_wikitext_finetune_qasrl_modal.py --config-name small
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium
```

**Record:** Final validation F1 for each size.

If a model can't achieve >20% F1 with full finetuning, it won't show emergence.

---

### Phase 2: Test for Emergence (Linear Probe)

**Goal:** Test if pretrained representations already encode semantic roles.

```bash
# Run with frozen LLM for sizes that passed Phase 1
# Use --skip-pretrain and --pretrained-model-path to reuse pretrained models from Phase 1
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm --skip-pretrain \
  --pretrained-model-path "<TinyDir>/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small --freeze-llm --skip-pretrain \
  --pretrained-model-path "<SmallDir>/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm --skip-pretrain \
  --pretrained-model-path "<BaseDir>/toy_llm_unified_pretrained.pth"
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium --freeze-llm --skip-pretrain \
  --pretrained-model-path "<MediumDir>/toy_llm_unified_pretrained.pth"
```

**Record:** Final validation F1 for each size.

**Interpretation:**

| Frozen F1 vs Full F1 | Interpretation |
|---------------------|----------------|
| Frozen ~= Full | Strong emergence - capability already encoded |
| Frozen << Full | No emergence - capability learned during finetuning |
| Frozen increases with scale | Scale-dependent emergence |

---

### Phase 3: Few-Shot Learning Curve

**Goal:** Measure sample efficiency - how quickly does capability emerge with minimal supervision?

```bash
# Test with varying amounts of finetuning data
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --finetune-samples 10
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --finetune-samples 50
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --finetune-samples 100
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --finetune-samples 500
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base  # Full dataset
```

**Record:** F1 at each sample count.

**Expected emergence signature:** Rapid performance gain with few examples (steep learning curve).

---

### Phase 4: Pretraining Data Ablation

**Goal:** Test if QA-format data in pretraining "leaks" the task.

Use the `--pretrain-data` flag to compare WikiText-only vs WikiText+NQ pretraining:

```bash
# Default: WikiText-103 only (clean emergence test)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base
# Then frozen experiment using that pretrained model:
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm --skip-pretrain \
  --pretrained-model-path "<BaseWikiDir>/toy_llm_unified_pretrained.pth"

# Ablation: Add nq_open to pretraining (may leak QA patterns)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --pretrain-data wiki+nq
# Then frozen experiment using that pretrained model:
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm --skip-pretrain \
  --pretrained-model-path "<BaseWikiNQDir>/toy_llm_unified_pretrained.pth"
```

**Compare:** Does adding nq_open help frozen LLM performance?
- If yes: nq_open was teaching QA patterns (not pure emergence from semantics)
- If no: WikiText alone provides sufficient semantic structure for emergence

---

## Full Experimental Matrix

| Model | Pretrain Data | Finetune Mode | Finetune Samples | Experiment ID |
|-------|--------------|---------------|------------------|---------------|
| tiny | Wiki + NQ | Full | All | T-WN-F-ALL |
| tiny | Wiki + NQ | Frozen | All | T-WN-Z-ALL |
| tiny | Wiki only | Full | All | T-W-F-ALL |
| tiny | Wiki only | Frozen | All | T-W-Z-ALL |
| small | Wiki + NQ | Full | All | S-WN-F-ALL |
| small | Wiki + NQ | Frozen | All | S-WN-Z-ALL |
| small | Wiki only | Full | All | S-W-F-ALL |
| small | Wiki only | Frozen | All | S-W-Z-ALL |
| base | Wiki + NQ | Full | All | B-WN-F-ALL |
| base | Wiki + NQ | Frozen | All | B-WN-Z-ALL |
| base | Wiki + NQ | Full | 100 | B-WN-F-100 |
| base | Wiki + NQ | Frozen | 100 | B-WN-Z-100 |
| base | Wiki only | Full | All | B-W-F-ALL |
| base | Wiki only | Frozen | All | B-W-Z-ALL |
| medium | Wiki + NQ | Full | All | M-WN-F-ALL |
| medium | Wiki + NQ | Frozen | All | M-WN-Z-ALL |
| medium | Wiki only | Full | All | M-W-F-ALL |
| medium | Wiki only | Frozen | All | M-W-Z-ALL |

---

## Metrics to Collect

For each experiment, record:

1. **Validation F1** (primary metric)
2. **Exact Match accuracy**
3. **Training loss curve**
4. **Number of trainable parameters** (verify frozen mode)
5. **Training time** (to measure efficiency)

---

## Expected Results & Interpretation

### Scenario A: Strong Emergence
```
Model   | Full F1 | Frozen F1
--------|---------|----------
tiny    | 25%     | 5%
small   | 45%     | 15%
base    | 70%     | 65%   <-- Jump!
medium  | 75%     | 73%
```
**Interpretation:** Emergence threshold at base scale. Semantic roles are encoded in pretrained representations.

### Scenario B: No Emergence
```
Model   | Full F1 | Frozen F1
--------|---------|----------
tiny    | 25%     | 5%
small   | 45%     | 8%
base    | 70%     | 12%
medium  | 75%     | 15%
```
**Interpretation:** Finetuning is required. The capability is learned, not emergent.

### Scenario C: Scale-Dependent Transfer
```
Model   | Full F1 | Frozen F1
--------|---------|----------
tiny    | 25%     | 5%
small   | 45%     | 20%
base    | 70%     | 40%
medium  | 75%     | 55%
```
**Interpretation:** Gradual improvement - partial encoding that improves with scale.

---

## Command Reference

```bash
# Full training (pretrain on WikiText + finetune on QA-SRL/SQuAD)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size>

# Use WikiText + NQ for pretraining (ablation study)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --pretrain-data wiki+nq

# Skip pretraining (use existing pretrained model from current run dir)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --skip-pretrain

# Frozen LLM with external pretrained model (recommended for linear probe experiments)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --freeze-llm --skip-pretrain \
  --pretrained-model-path "<ModelDir>/toy_llm_unified_pretrained.pth"

# Few-shot finetuning
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --finetune-samples 100

# Combined: frozen + few-shot
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --freeze-llm --skip-pretrain \
  --pretrained-model-path "<ModelDir>/toy_llm_unified_pretrained.pth" --finetune-samples 100

# Random baseline (establishes performance floor, skips pretraining)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --random-baseline --skip-pretrain

# List saved models
modal run pretrain_wikitext_finetune_qasrl_modal.py --list-saved
```

### Available Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--config-name` | tiny, small, base, medium | tiny | Model size configuration |
| `--pretrain-data` | wiki, wiki+nq | wiki | Pretraining data. Use `wiki` for clean emergence testing |
| `--freeze-llm` | (flag) | False | Freeze LLM weights, train only QA head (linear probe) |
| `--random-baseline` | (flag) | False | Use random init (use with --skip-pretrain) for emergence floor |
| `--finetune-samples` | integer | all | Limit finetuning samples for few-shot experiments |
| `--skip-pretrain` | (flag) | False | Skip pretraining phase |
| `--pretrained-model-path` | path | None | Path to pretrained model (use with --skip-pretrain --freeze-llm) |
| `--skip-finetune` | (flag) | False | Skip finetuning phase |
| `--list-saved` | (flag) | False | List all saved models |

---

## Notes

- **Model sizes:** tiny (~500K), small (~2M), base (~50M), medium (~100M+)
- **QA-SRL questions:** "Who did something?", "What did someone do?", etc.
- **Default pretraining data:** WikiText-103 only (no QA-format data for clean emergence testing)
- **Optional pretraining data:** `--pretrain-data wiki+nq` adds nq_open (QA pairs) - use for ablation only
- **The emergence hypothesis:** Next-token prediction on rich text implicitly teaches predicate-argument structure
- **GPU:** Runs on A10G by default (configured in script)