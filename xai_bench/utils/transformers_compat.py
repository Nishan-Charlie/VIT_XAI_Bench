"""Compatibility shim for LXT under transformers >= 5.

transformers v5 removed ``transformers.pytorch_utils.find_pruneable_heads_and_indices``,
but LXT's ``lxt/efficient/models/bert.py`` still imports it at module scope, and
``lxt.efficient.core`` imports that module eagerly. Importing the LXT backend then
raises ImportError under transformers >= 5, although the symbol belongs to a BERT
head-pruning path that no vision model reaches.

Importing this module restores the symbol with the original v4 implementation
(not a stub), so behaviour is unchanged should anything call it.
"""
import torch


def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Behaviour of the transformers v4 utility of the same name: the heads that
    can still be pruned (already-pruned ones excluded, reindexed) and the row
    indices of the weight matrix to keep."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def apply() -> bool:
    """Install the shim if the installed transformers lacks the symbol."""
    try:
        from transformers import pytorch_utils
    except Exception:  # noqa: BLE001 - transformers absent: nothing to patch
        return False
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
        return True
    return False


applied = apply()
