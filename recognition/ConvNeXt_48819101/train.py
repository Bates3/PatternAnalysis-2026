import random
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modules import ResNet
from predict import evaluate

torch.backends.cudnn.enabled = False
EPOCHS = 40
LR = 3e-4
WD = 0.05
LABEL_SMOOTHING = 0.1   
BATCH=64
SEED=123
CUTOFF = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH)

    model = ResNet(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    evaluation_criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")


    opt = torch.optim.AdamW(model.parameters(), weight_decay=WD, lr=LR)
    random.seed(123)
    torch.manual_seed(123)

    best_auc, best_epoch = -1.0, 0
    total = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)



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
            sched.step()

            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += ((logits > 0).long() == y).sum().item()
        val = evaluate(model, val_loader)
        v = val["slice"]
        print(f"epoch: {epoch+1} | train loss {run_loss/seen} acc: {correct/seen}")
        print(v)

        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()
