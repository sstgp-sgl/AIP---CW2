"""Global classifier plus a lesion head trained with the binary mask.

The global head matches the baseline. The lesion head pools only the predicted
lesion. Rejection can use agreement between the two heads.
"""
