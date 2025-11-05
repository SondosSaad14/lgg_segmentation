import torch
from typing import Dict

@torch.no_grad()
def seg_counts(logits, targets, thr: float=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).to(targets.dtype)
    y = targets.to(targets.dtype)
    tp = (pred*y).sum().item()
    fp = (pred*(1-y)).sum().item()
    fn = ((1-pred)*y).sum().item()
    tn = ((1-pred)*(1-y)).sum().item()
    total = y.numel()
    return tp, fp, fn, tn, total


@torch.no_grad()
def seg_metrics(tp, fp, fn, tn, total) -> Dict[str,float]:
    eps = 1e-12
    acc = (tp+tn)/(total+eps)
    prec = tp/(tp+fp+eps)
    rec = tp/(tp+fn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    iou = tp/(tp+fp+fn+eps)
    dice = 2*tp/(2*tp+fp+fn+eps)
    return {"accuracy":float(acc), "precision":float(prec), "recall":float(rec), "f1":float(f1), "iou":float(iou), "dice":float(dice)}
