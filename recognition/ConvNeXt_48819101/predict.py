import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

CUTOFF = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def evaluate(model, data, criterion=None):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    model.eval()
    probs = []
    labels = []
    loss_sum, n = 0.0, 0
    for batch in data:
        x, y = batch[0].to(device), batch[1].to(device)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y.float())

        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()

    pred = (probs >= CUTOFF).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
    metrics = {"acc": (tp+tn)/len(labels), "ad_recall": tp/(tp + fn), 
               "nc_recall": tn/(tn+fp), "auc": roc_auc_score(labels, probs),
               "fn_rate": fn/(tp+fn)}
    return {"loss": loss_sum / n, "slice": metrics}