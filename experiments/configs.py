# experiments/configs.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/raw"
    train_subdir: str = "train"
    val_subdir: str = "val"
    num_classes: int = 3
    class_names: Optional[list] = None

    # model
    model_name: str = "resnet18"     # timm-compatible name
    pretrained: bool = True

    # training
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"         # or "SGD"
    scheduler: str = "StepLR"
    step_size: int = 5
    gamma: float = 0.1

    # freeze / unfreeze
    freeze_backbone: bool = True
    unfreeze_at_epoch: int = 5

    # reproducibility + device
    seed: int = 42
    device: str = "cuda"             # or "cpu"

    # logging / paths
    out_dir: str = "models"
    logs_dir: str = "experiments/logs"
    num_workers: int = 4
