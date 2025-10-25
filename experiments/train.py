# experiments/train.py
import os
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from datasets import ImageFolderSimple, get_transforms
from utils import set_seed


# -----------------------------
# Utility functions
# -----------------------------
def freeze_backbone(model):
    """Замораживает все параметры кроме head/классификатора"""
    for name, param in model.named_parameters():
        if 'head' in name or 'classifier' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def compute_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Training Function
# -----------------------------
def train(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    # Prepare data
    train_t, val_t = get_transforms(cfg["image_size"])
    train_ds = ImageFolderSimple(os.path.join(cfg["data_dir"], cfg["train_subdir"]), transform=train_t)
    val_ds = ImageFolderSimple(os.path.join(cfg["data_dir"], cfg["val_subdir"]),
                               transform=val_t, classes=list(train_ds.class_to_idx.keys()))

    class_names = list(train_ds.class_to_idx.keys())
    num_classes = len(class_names)
    cfg["num_classes"] = num_classes
    cfg["class_names"] = class_names

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    # Model
    model = timm.create_model(cfg["model_name"], pretrained=cfg["pretrained"], num_classes=num_classes)
    model.to(device)
    if cfg["freeze_backbone"]:
        freeze_backbone(model)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
    criterion = nn.CrossEntropyLoss()

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(cfg["logs_dir"], exist_ok=True)
    config_path = os.path.join(cfg["logs_dir"], f"config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"✅ Config saved to {config_path}")

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, vcorrect, vtotal = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss /= vtotal
        val_acc = vcorrect / vtotal

        print(f"\nEpoch {epoch}/{cfg['epochs']}: "
              f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(cfg["out_dir"], exist_ok=True)
            best_model_path = os.path.join(cfg["out_dir"], f"best_{cfg['model_name']}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Saved best model to {best_model_path}")

        # Unfreeze backbone
        if cfg["freeze_backbone"] and epoch == cfg["unfreeze_at_epoch"]:
            for p in model.parameters():
                p.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] = cfg["lr"] * 0.1
            print(f"🔓 Unfroze backbone at epoch {epoch}, decreased LR ×0.1")

        scheduler.step()

        # Save confusion matrix
        cm_path = os.path.join(cfg["logs_dir"], f"cm_epoch{epoch}_{timestamp}.png")
        compute_confusion_matrix(all_labels, all_preds, class_names, cm_path)
        print(f"📊 Saved confusion matrix: {cm_path}")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

    print(f"\n✅ Training finished! Best val acc: {best_val_acc:.4f}")


# -----------------------------
# CLI parser
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning pretrained models")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--val_subdir", type=str, default="val")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    parser.add_argument("--unfreeze_at_epoch", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--scheduler", type=str, default="StepLR")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--logs_dir", type=str, default="experiments/logs")
    parser.add_argument("--num_workers", type=int, default=4)

    args = vars(parser.parse_args())
    train(args)
