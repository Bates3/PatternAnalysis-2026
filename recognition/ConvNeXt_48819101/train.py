import random
import torch
import torch.nn as nn
import math

from dataset import get_dataloaders
from modules import ResNet
from modules import ConvNeXt
from predict import evaluate


EPOCHS = 40
WARMUP_EPOCHS = 4
LR = 3e-4
WD = 0.05
LABEL_SMOOTHING = 0.1   
BATCH=64
SEED=123
CUTOFF = 0.5
DROP_PATH = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"


def main(model_name):
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH)
    if model_name == 'resnet50':
        model = ResNet(num_classes=1, drop_path_rate=DROP_PATH).to(device)
        torch.backends.cudnn.enabled = False # Fixes bug with running on AMD rocm
    elif model_name == 'convnext':
        model = ConvNeXt(num_classes=1, drop_path_rate=DROP_PATH).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # no weight decay on normal paramters
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.999),
    )


    steps_per_epoch = len(train_loader)
    total = EPOCHS * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch
    
    # linear warmup, then cosine decay
    def lr_lambda(step):                            
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    on_gpu = device=="cuda"
    if (on_gpu):
        print("YAY!")
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)

    

    best_auc, best_epoch = -1.0, 0
    total = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)



    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0
        seen = 0
        correct = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            y = y.float()
            y_smooth = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on_gpu):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y_smooth)

            scaler.scale(loss).backward()
            # scaler.unscale_(opt)
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += ((logits > 0).long() == y.long()).sum().item()
        val = evaluate(model, val_loader)
        v = val["slice"]
        print(f"epoch: {epoch+1} | train loss {run_loss/seen} acc: {correct/seen}")
        print(v)

        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{model_name}.pt")
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    print(f"best val auc {best_auc:.4f} @ epoch {best_epoch}")
    print("test:", evaluate(model, test_loader)["slice"])

if __name__ == "__main__":
    # main('resnet50')
    main('convnext')
