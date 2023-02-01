import torch
import numpy as np
def sensitivity(preds, target):
    target=target.cpu().numpy()
    tp = (preds*target).sum()
    fn =((1 - preds) * target).sum()
    return tp / (tp + fn)

