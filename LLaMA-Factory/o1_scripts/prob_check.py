from transformers import AutoModelForCausalLM, AutoTokenizer

def get_gated_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the gated log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the gated sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    
    # Compute per-token probabilities and log probabilities
    with torch.no_grad():
        per_token_probs = logits.softmax(-1).gather(dim=2, index=labels.unsqueeze(2)).squeeze(2)

    per_token_logps = logits.log_softmax(-1).gather(dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # Compute gating factor k for each sequence
    valid_token_count = loss_mask.sum(-1)
    gated_denominator = valid_token_count - (per_token_probs * loss_mask).sum(-1)

    print("per_token_probs:",per_token_probs*loss_mask)

    with torch.no_grad():
        k = valid_token_count / gated_denominator
        # print("valid_token_count:",valid_token_count,"gated_denominator:",gated_denominator,"k:",k)
    # Apply gating to per-token log probabilities
    alpha = 1
    gated_logps = (1 - alpha) * per_token_logps + alpha * k.unsqueeze(1) * (1 - per_token_probs) * per_token_logps

    # gated_logps = k.unsqueeze(1) * (1 - per_token_probs) * per_token_logps

    # Compute final log probabilities and valid lengths
    total_logps = (gated_logps * loss_mask).sum(-1)
    valid_length = valid_token_count

    return total_logps, valid_length