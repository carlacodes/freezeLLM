import torch
from sklearn.metrics import f1_score


def evaluate_qa_metrics(model, dataloader, device):
    model.eval()
    all_pred_start, all_pred_end, all_true_start, all_true_end = [], [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, start_pos, end_pos in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            start_pos, end_pos = start_pos.to(device), end_pos.to(device)
            _, start_logits, end_logits = model(
                input_ids, attention_mask=attention_mask
            )
            all_pred_start.append(torch.argmax(start_logits, dim=1).cpu())
            all_pred_end.append(torch.argmax(end_logits, dim=1).cpu())
            all_true_start.append(start_pos.cpu())
            all_true_end.append(end_pos.cpu())
    all_pred_start = torch.cat(all_pred_start)
    all_pred_end = torch.cat(all_pred_end)
    all_true_start = torch.cat(all_true_start)
    all_true_end = torch.cat(all_true_end)
    exact_match = (
        ((all_pred_start == all_true_start) & (all_pred_end == all_true_end))
        .float()
        .mean()
        .item()
    )
    f1_start = f1_score(all_true_start.numpy(), all_pred_start.numpy(), average="macro")
    f1_end = f1_score(all_true_end.numpy(), all_pred_end.numpy(), average="macro")
    f1 = (f1_start + f1_end) / 2
    return exact_match, f1
