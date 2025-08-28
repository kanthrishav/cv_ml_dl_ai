#!/usr/bin/env python3
# train_and_export_pytorch.py
# PyTorch fine-tune MobileNetV2 and export TorchScript for Raspberry Pi inference.
# NO CLI args; configure constants below.

import os, time, math, copy, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ========================== CONFIG (EDIT HERE) ==========================
DATA_DIR     = os.path.join("..", "..", "..", "data", "openimages")         # produced by openimages_to_classification.py
EXPORT_DIR   = "./export_torch"            # where model_ts.pt and labels.txt are saved
IMG_SIZE     = 224
BATCH_SIZE   = 64                          # will auto-reduce on OOM
EPOCHS_HEAD  = 6                           # frozen backbone
EPOCHS_FT    = 2                           # light fine-tune last layers
UNFREEZE_LAST_LAYERS = 20                  # unfreeze last N layers of features
NUM_WORKERS  = 4
LR_HEAD      = 1e-3
LR_FT        = 1e-4
WEIGHT_DECAY = 1e-4
SEED         = 42
TARGET_GPU_MEM_FRACTION = 0.85             # cap allocator near 85% of VRAM
USE_AMP      = True                        # mixed precision to fit batch + speed up
# ======================================================================

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

def cap_allocator(fraction=0.85):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            print(f"[INFO] CUDA memory fraction capped at {fraction:.2f}")
        except Exception as e:
            print(f"[WARN] set_per_process_memory_fraction not available: {e}")

def make_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf

def build_loaders():
    train_tf, val_tf = make_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
    class_names = train_ds.classes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, class_names

def build_model(num_classes):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    base = models.mobilenet_v2(weights=weights)
    for p in base.features.parameters():
        p.requires_grad = False
    # Replace classifier head
    in_feats = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_feats, num_classes)
    return base

def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    running_loss = 0.0; running_acc = 0.0; n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs); loss = criterion(out, labels)
            loss.backward(); optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc  += accuracy(out, labels) * imgs.size(0)
        n += imgs.size(0)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0; acc_sum = 0.0; n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        acc_sum  += accuracy(out, labels) * imgs.size(0)
        n += imgs.size(0)
    return loss_sum / n, acc_sum / n

def try_train_with_batch(model, train_loader, val_loader, class_names, device):
    global BATCH_SIZE
    criterion = nn.CrossEntropyLoss()
    # HEAD training
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)

    best_acc = 0.0; best_state = None
    for epoch in range(EPOCHS_HEAD):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=USE_AMP)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[HEAD] Epoch {epoch+1}/{EPOCHS_HEAD} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # UNFREEZE last layers
    unfrozen = 0
    for m in list(model.features.children())[-UNFREEZE_LAST_LAYERS:]:
        for p in m.parameters():
            p.requires_grad = True; unfrozen += 1
    print(f"[INFO] Unfroze params in last {UNFREEZE_LAST_LAYERS} feature blocks (params toggled: {unfrozen})")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR_FT, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=USE_AMP)

    for epoch in range(EPOCHS_FT):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=USE_AMP)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[FT ] Epoch {epoch+1}/{EPOCHS_FT} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, class_names

def export_torchscript(model, class_names):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model.eval().cpu()
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        ts = torch.jit.trace(model, example)  # trace is fine for MobilenetV2 classifier
    ts_path = os.path.join(EXPORT_DIR, "model_ts.pt")
    ts.save(ts_path)
    print(f"[INFO] TorchScript saved -> {ts_path}")

    # Write labels.txt in the order of dataset classes
    with open(os.path.join(EXPORT_DIR, "labels.txt"), "w") as f:
        for c in class_names:
            f.write(c + "\n")
    print(f"[INFO] labels.txt saved -> {os.path.join(EXPORT_DIR, 'labels.txt')}")

def main():
    cap_allocator(TARGET_GPU_MEM_FRACTION)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build loaders; on OOM reduce batch size progressively
    global BATCH_SIZE
    while True:
        try:
            train_loader, val_loader, class_names = build_loaders()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 16:
                BATCH_SIZE = max(16, BATCH_SIZE // 2)
                print(f"[WARN] OOM at DataLoader creation; reducing BATCH_SIZE to {BATCH_SIZE} and retrying.")
            else:
                raise

    model = build_model(num_classes=len(class_names)).to(device)

    # Train with OOM-protected loop (will downshift batch if needed)
    trained = False
    while not trained:
        try:
            model, class_names = try_train_with_batch(model, train_loader, val_loader, class_names, device)
            trained = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 16:
                BATCH_SIZE = max(16, BATCH_SIZE // 2)
                print(f"[WARN] OOM during training; reducing BATCH_SIZE to {BATCH_SIZE} and rebuilding loaders.")
                train_loader, val_loader, class_names = build_loaders()
            else:
                raise

    # Export TorchScript
    export_torchscript(model, class_names)

if __name__ == "__main__":
    main()
