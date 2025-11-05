import os, random
from typing import Tuple, Dict, List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def _patient_id_from(dirpath: str, root: str) -> str:
    rel = os.path.relpath(dirpath, root)
    parts = [p for p in rel.split(os.sep) if p not in (".", "")]
    return parts[0] if parts else os.path.basename(dirpath)

def _normalize_stem(basename: str) -> str:
    stem = os.path.splitext(basename)[0].lower()
    tokens = [
        "_mask", "-mask", " mask",
        "_seg", "-seg", " seg",
        "_segmentation", " segmentation",
        "_gt", "-gt", " groundtruth", " ground_truth",
        "_annotation", "-annotation"
    ]
    for t in tokens:
        stem = stem.replace(t, "")
    stem = stem.replace("-", "_").replace(" ", "")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem

def list_lgg_pairs(root: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data root not found: {root}")

    images, masks = {}, {}
    for dirpath, _, files in os.walk(root):
        rel_parts = os.path.relpath(dirpath, root).split(os.sep)
        in_mask_dir = any(p.lower() in ("mask", "masks") for p in rel_parts)
        for f in files:
            if not is_image(f):
                continue
            full = os.path.join(dirpath, f)
            pid = _patient_id_from(dirpath, root)
            base = os.path.basename(f)
            is_mask_file = ("mask" in base.lower()) or ("seg" in base.lower()) or ("gt" in base.lower()) or ("annot" in base.lower()) or in_mask_dir
            key = (pid, _normalize_stem(base))
            if is_mask_file:
                masks.setdefault(key, []).append(full)
            else:
                images.setdefault(key, []).append(full)

    pairs = []
    for key in sorted(set(images.keys()) & set(masks.keys())):
        pairs.append((images[key][0], masks[key][0], key[0]))

    if not pairs:
        candidates = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if is_image(f) and ("mask" in f.lower() or "seg" in f.lower() or "gt" in f.lower() or "annot" in f.lower() or "mask" in dirpath.lower() or "masks" in dirpath.lower()):
                    candidates.append(os.path.join(dirpath, f))
        print(f"[Diagnostics] Found {len(candidates)} mask-like files.")
        for p in candidates[:10]:
            print("  -", os.path.relpath(p, root))
        raise RuntimeError("No (image, mask) pairs found. Review naming patterns or folder structure.")

    return pairs

def split_by_patient(pairs, val_ratio: float, test_ratio: float, seed: int=42):
    patients = sorted({pid for _,_,pid in pairs})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_p = set(patients[:n_train])
    val_p   = set(patients[n_train:n_train+n_val])
    test_p  = set(patients[n_train+n_val:])
    tr = [p for p in pairs if p[2] in train_p]
    va = [p for p in pairs if p[2] in val_p]
    te = [p for p in pairs if p[2] in test_p]
    return tr, va, te, patients

class LGGSegmentationDataset(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    def __init__(self, pairs, img_size, augment=True, imagenet_norm=True):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.imagenet_norm = imagenet_norm

    def __len__(self): return len(self.pairs)

    def _load(self, ip, mp):
        img = Image.open(ip).convert("RGB")
        msk = Image.open(mp).convert("L")
        return img, msk

    def _resize(self, img, msk):
        H,W = self.img_size
        img = TF.resize(img, size=[H,W], interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.resize(msk, size=[H,W], interpolation=TF.InterpolationMode.NEAREST)
        return img, msk

    def _augment(self, img, msk):
        if random.random() < 0.5:
            img = TF.hflip(img); msk = TF.hflip(msk)
        angle = random.uniform(-20, 20)
        translate = (int(random.uniform(-0.05,0.05)*img.width),
                     int(random.uniform(-0.05,0.05)*img.height))
        scale = random.uniform(0.95, 1.05)
        shear = random.uniform(-5, 5)
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[shear,0.0],
                        interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.affine(msk, angle=angle, translate=translate, scale=scale, shear=[shear,0.0],
                        interpolation=TF.InterpolationMode.NEAREST)
        return img, msk

    def __getitem__(self, idx):
        ip, mp, pid = self.pairs[idx]
        img, msk = self._load(ip, mp)
        img, msk = self._resize(img, msk)
        if self.augment:
            img, msk = self._augment(img, msk)
        img_t = TF.to_tensor(img)
        if self.imagenet_norm:
            img_t = TF.normalize(img_t, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        msk_t = torch.from_numpy(np.array(msk, dtype=np.uint8))
        msk_t = (msk_t > 127).float().unsqueeze(0)
        return img_t, msk_t, ip, mp, pid

def make_dataloaders(cfg):
    pairs = list_lgg_pairs(cfg.data_dir)
    tr, va, te, patients = split_by_patient(pairs, cfg.val_split, cfg.test_split, cfg.seed)
    print(f"Patients: {len(patients)}")
    print(f"Pairs: train={len(tr)} | val={len(va)} | test={len(te)}")

    pin = cfg.pin_memory and (cfg.device == "cuda")
    train_ds = LGGSegmentationDataset(tr, cfg.img_size, augment=True,  imagenet_norm=cfg.imagenet_norm)
    val_ds   = LGGSegmentationDataset(va, cfg.img_size, augment=False, imagenet_norm=cfg.imagenet_norm)
    test_ds  = LGGSegmentationDataset(te, cfg.img_size, augment=False, imagenet_norm=cfg.imagenet_norm)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
