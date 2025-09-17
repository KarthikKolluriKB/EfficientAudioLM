import torch

def compute_accuracy(pad_outputs: torch.LongTensor,
                     pad_targets: torch.LongTensor,
                     ignore_label: int, 
                     return_counts: bool = False) -> float:
    """
    Token Accuracy with masking.

    Args:
        pad_outputs: [B, T] predicted token ids.
        pad_targets: [B, T] target token ids.
        ignore_label: label value to ignore (e.g., -100).
        return_counts: if True, also return (num_correct, num_total) to aggregate.

    Returns:
        acc : accuracy value (float) in [0, 1].
        (optional) (num_correct, num_total): counts of correct and total tokens.
    """
    # pad_outputs, pad_targets: [B, T]
    mask = pad_targets != ignore_label
    denom = mask.sum()

    if denom.item() == 0:
        acc = pad_outputs.new_tensor(0.0)
        return (acc, 0, 0) if return_counts else acc
    
    num_correct = (pad_outputs.eq(pad_targets) & mask).sum()
    acc = num_correct.float() / denom.float()
    return (acc, num_correct.item(), denom.item()) if return_counts else acc


def next_token_accuracy_from_logits(logits: torch.FloatTensor,
                       labels: torch.LongTensor,
                       ignore_label: int = -100,
                       return_counts: bool = False) -> float:
    """
    Align predictions/targets for next-token accuracy.

    Compares the prediction at time t with the target at time t+1.
    """
    # logits: [B, T, V]
    # labels: [B, T]
    preds = logits.argmax(dim=-1)  # [B, T] 
    preds = preds[:, :-1]          # [B, T-1]
    targets = labels[:, 1:]        # [B, T-1]
    return compute_accuracy(preds, targets, ignore_label, return_counts)


# TODO: WER computation Metric