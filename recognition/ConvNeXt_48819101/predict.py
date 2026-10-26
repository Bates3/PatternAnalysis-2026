import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

from modules import ResNet
from dataset import get_dataloaders
device = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def evaluate(model, data, criterion=None, cutoff=0.5, plot=False):
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

    pred = (probs >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
    metrics = {"acc": (tp+tn)/len(labels), "ad_recall": tp/(tp + fn), 
               "nc_recall": tn/(tn+fp), "auc": roc_auc_score(labels, probs),
               "fn_rate": fn/(tp+fn)}
    print(f"---------------CUTOFF = {cutoff} ---------------------")
    if plot:
        plot_roc(probs, labels, np.arange(0.3, 0.6, 0.05), save_path="roc.png")
    return {"loss": loss_sum / n, "slice": metrics}


def plot_roc(probs, labels, cutoffs, save_path=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    for c in cutoffs:
        pred = (probs >= c).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        x, y = fp / max(fp + tn, 1), tp / max(tp + fn, 1)
        ax.scatter(x, y, s=40, zorder=3)
        ax.annotate(f"{c:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)

    ax.set_xlabel("False positive rate (1 - nc_recall)")
    ax.set_ylabel("True positive rate (ad_recall)")
    ax.set_title("ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main(ckpt="resnet50.pt"):
    plot = False
    _, _, test_loader = get_dataloaders()
    model = ResNet(num_classes=1).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    res = evaluate(model, test_loader)
    for i in np.arange(0.3, 0.6, 0.05):
        if i > 0.54:
            plot = True
        result = evaluate(model, test_loader, cutoff=i, plot=plot)
        print(f"test loss: {result['loss']:.4f}")
        print({k: round(float(v), 4) for k, v in result["slice"].items()})
    

if __name__ == "__main__":
    main()