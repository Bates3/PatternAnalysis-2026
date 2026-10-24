import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

from dataset import get_dataloaders
from modules import ResNet



EPOCHS = 40
LR = 3e-4
WD = 0.05
LABEL_SMOOTHING = 0.1   
BATCH=64
SEED=123
CUTOFF = 0.5
device = "cuda" if torch.cuda_is_available() else "cpu"

def evaluate(model, data, criterion):
    model.eval()
    probs = []
    labels = []
    for batch in data:
        x, y = batch[0].to(device), batch[1].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()

    pred = (probs >= CUTOFF).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    metrics = {"acc": (tp+tn)/len(labels), "ad_recall": tp/(tp + fn), 
               "nc_recall": tn/(tn+fp), "auc": roc_auc_score(labels, pred),
               "fn_rate": fn/(tp+fn)}
    return {"loss": loss_sum / n, "slice": metrics}


def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH)

    model = ResNet(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    evaluation_criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")


    opt = torch.optim.AdamW(weight_decay=WD, lr=LR)

    decay = []
    no_decay = []

    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0
        seen = 0
        correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            logits = model(x).squeeze(1)
            loss = criterion(logits, y.float())

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scaler.step(opt)

            run_loss += loss.item() * y.size(0)
            seen += y.size()
            correct += ((logits > 0).long() == y).sum().item()
        val = evaluate(model, val_loader)
        print(f"epoch: {epoch+1} | train loss {run_loss/seen} acc: {correct/seen}")
    