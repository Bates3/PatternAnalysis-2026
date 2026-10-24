import random
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modules import ResNet
from predict import evaluate


EPOCHS = 40
LR = 3e-4
WD = 0.05
LABEL_SMOOTHING = 0.1   
BATCH=64
SEED=123
CUTOFF = 0.5
device = "cuda" if torch.cuda_is_available() else "cpu"


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
    