"""Lesion-aware pooling for AIP304 CW2 Option 2.

The 7x7 decoder still predicts a binary lesion map. Classification pools a
residual reweighting of the last feature map by that prediction. Ground-truth
masks are used only in the training loss.
"""
