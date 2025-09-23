import torch
from typing import List, Tuple
from jiwer import RemoveWhiteSpace, wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords

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

# TODO: preset for each dataset (e.g., Librispeech, CommonVoice, etc.) can be configured
def compute_wer(predictions, references) -> float:
    """
    Computes Average Word Error Rate (WER) for single or batch predictions and references.

    Args:
        predictions: predicted string or list of predicted strings.
        references: reference string or list of reference strings.

    Returns:
        Average WER (float).
    """
    # Normalize to list
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references), "Predictions and references must have the same length."

    # Use a Normalization pipeline for robust, fair WER scoring
    transformation = Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveWhiteSpace(replace_by_space=True),
        RemoveMultipleSpaces(),
        Strip(),
        ReduceToListOfListOfWords()  
    ])

    wer_score = wer(
        references,
        predictions,
        reference_transform=transformation,
        hypothesis_transform=transformation
    )

    return wer_score


def decode_texts_from_outputs(
    outputs, 
    labels, 
    tokenizer,
    ignore_index: int = -100, 
    shift_causal: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = None,
) -> Tuple[List[str], List[str]]: 
    """
    Decode text from model outputs and labels using the provided tokenizer.

    Args:
        outputs: model outputs (logits or token ids).
        labels: target token ids.
        tokenizer: tokenizer with a decode method.
        ignore_index: label value to ignore (e.g., -100).
        shift_causal: if True, shift outputs and labels for causal models.
        skip_special_tokens: if True, skip special tokens during decoding.
        clean_up_tokenization_spaces: if True, clean up tokenization spaces.

    Returns:
        hyp_texts (List[str]): list of decoded hypothesis texts.
        ref_texts (List[str]): list of decoded reference texts.
    """
    # logits: [B, T, V] B- batch size, T- seq len, V- vocab size
    logist = outputs.logits if hasattr(outputs, "logits") else outputs

    with torch.no_grad():
        pred_ids = logist.argmax(dim=-1)  # [B, T] batch of token ids

    # Align t predictions with t+1 labels for causal LM if requested
    if shift_causal:
        pred_ids = pred_ids[:, :-1]      # [B, T-1]
        ref_ids = labels[:, 1:]           # [B, T-1]
    else:
        ref_ids = labels  # [B, T]

    # Mask valid reference positions (ignore -100)
    mask = ref_ids.ne(ignore_index)  # [B, T] or [B, T-1]

    hyp_texts, ref_texts = [], []
    B = pred_ids.size(0) # batch size
    for b in range(B): 
        m = mask[b]
        if m.sum().item() == 0: 
            continue  # skip empty references
        hyp_token = pred_ids[b][m] # [T_valid]
        ref_token = ref_ids[b][m]   # [T_valid]
        # Decode hypothesis text (tokens to text)
        hyp_text = tokenizer.decode(
            hyp_token, 
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        # Decode reference text
        ref_text = tokenizer.decode(
            ref_token,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        hyp_texts.append(hyp_text)
        ref_texts.append(ref_text)

    return hyp_texts, ref_texts
