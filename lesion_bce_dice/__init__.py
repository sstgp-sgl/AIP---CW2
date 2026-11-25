"""Auxiliary lesion decoder with BCE and soft Dice mask loss.

Same classification path and 7x7 decoder as lesion_decoder. The mask term
is binary cross-entropy plus a per-image soft Dice loss. Evaluation still
uses classification logits only.
"""
