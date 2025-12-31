import torch


def evaluate_qa_metrics(model, dataloader, device):
    """
    Evaluates the model on a QA dataset using Exact Match and standard token-level F1 score.

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

            # The model forward pass returns (loss, start_logits, end_logits)
            # We don't need the loss during evaluation.
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

    # 1. --- Exact Match Calculation (remains unchanged) ---
    # This metric is correct and valuable. It measures the percentage of predictions
    # where both the start and end pointers are exactly correct.
    exact_match = (
        ((all_pred_start == all_true_start) & (all_pred_end == all_true_end))
        .float()
        .mean()
        .item()
    )

    # 2. --- Standard F1 Score Calculation (Token Overlap) ---
    # This metric gives partial credit for answers that are not an exact match
    # but still overlap significantly with the true answer.
    f1_scores = []
    for i in range(len(all_true_start)):
        true_start, true_end = all_true_start[i].item(), all_true_end[i].item()
        pred_start, pred_end = all_pred_start[i].item(), all_pred_end[i].item()

        # Create sets of token indices for the predicted and true answer spans.
        # Handle the case where the model predicts an invalid span (start > end).
        if pred_start > pred_end:
            pred_tokens = set()
        else:
            pred_tokens = set(range(pred_start, pred_end + 1))

        true_tokens = set(range(true_start, true_end + 1))

        # Calculate the number of common tokens between prediction and truth.
        common_tokens = len(pred_tokens.intersection(true_tokens))

        # If there are no common tokens, F1 is 0.
        if common_tokens == 0:
            f1_scores.append(0.0)
            continue

        # Calculate precision and recall.
        precision = common_tokens / len(pred_tokens)
        recall = common_tokens / len(true_tokens)

        # Calculate F1 score for this single example.
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    # The final F1 score is the average of the F1 scores across all examples.
    final_f1 = sum(f1_scores) / len(f1_scores)

    return exact_match, final_f1
