"""
Diagnostic script to verify QA-SRL data preprocessing alignment.
Checks that token positions correctly map back to the expected answer text.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def verify_alignment(num_samples=10):
    """Verify that start/end positions correctly identify the answer span."""

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_seq_len = 512

    print("Loading QA-SRL dataset...")
    dataset = load_dataset("qa_srl", split="validation", trust_remote_code=True)
    print(f"Loaded {len(dataset)} examples\n")

    misalignments = 0
    total_checked = 0

    for idx, example in enumerate(dataset):
        if total_checked >= num_samples:
            break

        context = example["sentence"]
        question = " ".join([token for token in example["question"] if token != "_"])

        if not example.get("answers"):
            continue

        for answer_text in example["answers"]:
            # This is how the preprocessing finds the answer
            char_start = context.lower().find(answer_text.lower())
            if char_start == -1:
                continue
            char_end = char_start + len(answer_text)

            encoding = tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_seq_len,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            for i in range(len(encoding["input_ids"])):
                sequence_ids = encoding.sequence_ids(i)
                context_indices = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]

                if not context_indices:
                    continue

                offset_mapping = encoding["offset_mapping"][i]
                token_start_index = -1
                token_end_index = -1

                for tok_idx in context_indices:
                    start, end = offset_mapping[tok_idx]
                    if start <= char_start < end:
                        token_start_index = tok_idx
                    if start < char_end <= end:
                        token_end_index = tok_idx

                if token_start_index != -1 and token_end_index != -1:
                    total_checked += 1

                    # Decode the predicted span
                    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][i])
                    predicted_tokens = tokens[token_start_index:token_end_index + 1]
                    predicted_text = tokenizer.convert_tokens_to_string(predicted_tokens)

                    # Check alignment
                    is_aligned = answer_text.lower().strip() == predicted_text.lower().strip()

                    print(f"{'='*60}")
                    print(f"Example {total_checked}")
                    print(f"{'='*60}")
                    print(f"Context: {context[:100]}...")
                    print(f"Question: {question}")
                    print(f"Expected answer: '{answer_text}'")
                    print(f"Decoded span:    '{predicted_text}'")
                    print(f"Token indices: start={token_start_index}, end={token_end_index}")
                    print(f"Tokens: {predicted_tokens}")

                    if is_aligned:
                        print(f"Status: ALIGNED")
                    else:
                        print(f"Status: MISALIGNED")
                        misalignments += 1

                        # Show more details for misalignments
                        print(f"\nDEBUG INFO:")
                        print(f"  char_start={char_start}, char_end={char_end}")
                        print(f"  Context at those positions: '{context[char_start:char_end]}'")
                        print(f"  Offset mapping at start token: {offset_mapping[token_start_index]}")
                        print(f"  Offset mapping at end token: {offset_mapping[token_end_index]}")

                    print()
                    break  # Only check first valid encoding

            break  # Only check first answer

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples checked: {total_checked}")
    print(f"Misalignments found: {misalignments}")
    print(f"Alignment rate: {100 * (total_checked - misalignments) / total_checked:.1f}%")

    if misalignments > 0:
        print(f"\nWARNING: Found {misalignments} misalignments!")
    else:
        print(f"\nAll examples correctly aligned.")


if __name__ == "__main__":
    verify_alignment(num_samples=20)