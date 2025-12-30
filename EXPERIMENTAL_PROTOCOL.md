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

Run these 4 commands to test emergence on the `tiny` model:

```bash
# 1. Baseline: Can tiny model learn the task at all?
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny

# 2. Linear probe: Does emergence occur?
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm

# 3. (Optional) Test if NQ data helps
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --pretrain-data wiki+nq
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --pretrain-data wiki+nq --freeze-llm
```

### Full Experiment (All Model Sizes)

```bash
# Phase 1: Baseline capability check (full finetuning)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium

# Phase 2: Test for emergence (linear probe / freeze LLM)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium --freeze-llm
```

### Listing Saved Models

```bash
modal run pretrain_wikitext_finetune_qasrl_modal.py --list-saved
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
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name tiny --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name small --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name medium --freeze-llm
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
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --freeze-llm

# Ablation: Add nq_open to pretraining (may leak QA patterns)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --pretrain-data wiki+nq
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name base --pretrain-data wiki+nq --freeze-llm
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

# Skip pretraining (use existing pretrained model)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --skip-pretrain

# Frozen LLM (linear probe)
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --freeze-llm

# Few-shot finetuning
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --finetune-samples 100

# Combined: frozen + few-shot
modal run pretrain_wikitext_finetune_qasrl_modal.py --config-name <size> --freeze-llm --finetune-samples 100

# List saved models
modal run pretrain_wikitext_finetune_qasrl_modal.py --list-saved
```

### Available Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--config-name` | tiny, small, base, medium | tiny | Model size configuration |
| `--pretrain-data` | wiki, wiki+nq | wiki | Pretraining data. Use `wiki` for clean emergence testing |
| `--freeze-llm` | (flag) | False | Freeze LLM weights, train only QA head (linear probe) |
| `--finetune-samples` | integer | all | Limit finetuning samples for few-shot experiments |
| `--skip-pretrain` | (flag) | False | Skip pretraining, use existing model |
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