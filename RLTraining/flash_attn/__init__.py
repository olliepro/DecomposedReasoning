"""Small compatibility namespace for Verl padding helpers.

The full `flash-attn` wheel is not available for the current Torch/CUDA stack on
Ascend. Verl only needs `flash_attn.bert_padding` for this RL path, so that
module is provided locally without exposing attention kernels.
"""
