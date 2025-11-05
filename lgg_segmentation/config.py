from dataclasses import dataclass, asdict
from typing import Tuple
import torch, os, random, numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    data_dir: str = "/kaggle/input/lgg-mri-segmentation/kaggle_3m"
    out_dir: str  = "artifacts_lgg_en"
    run_name: str = "lgg_multi_seg_en"

    img_size: Tuple[int,int] = (256, 256)
    val_split: float = 0.10
    test_split: float = 0.10
    num_workers: int = 2
    imagenet_norm: bool = True

    epochs: int = 40
    warmup_epochs_frozen: int = 4
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True

    def prepare(self):
        os.makedirs(self.out_dir, exist_ok=True)
        seed_everything(self.seed)
        return self
