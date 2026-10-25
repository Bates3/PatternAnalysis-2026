import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["NC", "AD"]  # index 0 = NC, index 1 = AD (AD is the "positive" class)
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ROOT = Path("data/ADNI/AD_NC")


def class_to_idx(cls):
    if cls == "NC":
        return 0
    return 1

def scan_split(split):
    """Transforms raw data files to a list of (filepath, class, id) tuples"""
    records = [] 
    for cls in CLASS_NAMES:
        folder = Path(ROOT) / split / cls          # e.g. AD_NC/test/AD
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for p in sorted((ROOT/split/cls).glob("*.jpeg")): # Gets files from location like AD_NC/test/AD
            patient_id, slice_num = p.stem.split('_')

            records.append((str(p), class_to_idx(cls), patient_id))
    if not records:
        raise FileNotFoundError(f"No images found under {ROOT/split}")
    return records


def split_by_patient(records, val_frac=0.5, seed=123):
    """Patient-level, class-stratified split into (val, test)."""
    patients_by_label = {0: [], 1: []}
    for path, label, pid in records:
        if pid not in patients_by_label[label]:
            patients_by_label[label].append(pid)

    rng = random.Random(seed)
    val_patients = []
    test_patients = []
    for label, patients in patients_by_label.items():
        rng.shuffle(patients)
        n = int(len(patients) * val_frac)
        val_patients = val_patients + patients[:n]
        test_patients = test_patients + patients[n:]
    val = []
    test = []
    for r in records:
        if r[2] in val_patients:
            val.append(r)
        if r[2] in test_patients:
            test.append(r)
    assert not ({r[2] for r in test} & {r[2] for r in val}), "Patient leakage: test/val"
    return test, val


class ADNIDataset(Dataset):
    """Transforms greyscale slices, resized, augmented (train only), 3-channel normalised tensor."""

    def __init__(self, records, img_size=224, train=False, return_path=False):
        self.records, self.return_path = records, return_path
        steps = [T.Resize((img_size, img_size))]
        if train:
            steps += [T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=9)]
        steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
        if train:
            steps += [T.RandomErasing(p=0.25)]
        self.tf = T.Compose(steps)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, label, _ = self.records[i]
        img = self.tf(Image.open(path).convert("RGB"))  # 1 channel replicated to 3
        return (img, label, path) if self.return_path else (img, label)


def _load_splits(val_frac=0.5, seed=123):
    train = scan_split("train")
    test_all = scan_split("test")
    test, val = split_by_patient(test_all, val_frac, seed)
    seen = {r[2] for r in train}
    assert not (seen & {r[2] for r in test}), "Patient leakage: train/test"
    assert not (seen & {r[2] for r in val}), "Patient leakage: train/test"
    return train, val, test


def get_dataloaders(root=None, batch_size=64, img_size=224, num_workers=3, val_frac=0.5, seed=123):
    """Returns (train_loader, val_loader, test_loader)."""
    train, val, test = _load_splits(val_frac, seed)
    for name, recs in (("train", train), ("val", val), ("test", test)):
        num_ad = sum(r[1] for r in recs)
        # Print a summary of data loaded
        print(f"{name}: {len(recs)} images, {len({r[2] for r in recs})} patients, "
              f"AD={num_ad} NC={len(recs) - num_ad}")
    return (
        DataLoader(ADNIDataset(train, img_size, train=True), batch_size, shuffle=True, drop_last=True, 
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(ADNIDataset(val, img_size), batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(ADNIDataset(test, img_size), batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    )