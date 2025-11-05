import os, json
from dataclasses import asdict
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import bce_dice_loss
from .metrics import seg_counts, seg_metrics
from .models import freeze_backbone, unfreeze_all, forward_logits

def train_eval_model(model, loaders, cfg, tag: str):
    train_loader, val_loader, test_loader = loaders
    device = cfg.device
    model.to(device)
    scaler = GradScaler(enabled=(cfg.mixed_precision and (device=='cuda')))

    freeze_backbone(model)
    head_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    head_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(head_opt, mode='min', factor=0.5, patience=3, verbose=True)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    best_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_{tag}_best.pt")


    def run_epoch(loader, train: bool):
        if train: model.train()
        else: model.eval()
        total_loss = 0.0

        tp=fp=fn=tn=total=0

        for imgs, msks, *_ in tqdm(loader, desc="Train" if train else "Eval", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with autocast(enabled=(cfg.mixed_precision and (device=='cuda'))):
                logits = forward_logits(model, imgs)
                if logits.shape[1] != 1: logits = logits[:, :1, ...]
                loss = bce_dice_loss(logits, msks)

            if train:
                head_opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(head_opt)
                scaler.update()
            total_loss += loss.item()
            a,b,c,d,t = seg_counts(logits, msks)
            tp+=a; fp+=b; fn+=c; tn+=d; total+=t
        return total_loss/len(loader), seg_metrics(tp,fp,fn,tn,total)


    for epoch in range(1, cfg.warmup_epochs_frozen+1):
        tr_loss, _ = run_epoch(train_loader, True)
        va_loss, va_m = run_epoch(val_loader, False)

        head_sch.step(va_loss)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_dice"].append(va_m["dice"])
        print(f"[{tag}] Warmup {epoch}/{cfg.warmup_epochs_frozen} | train={tr_loss:.4f} | val={va_loss:.4f} | Dice={va_m['dice']:.4f}")
        if va_m["dice"] > best_dice:
            best_dice = va_m["dice"]
            torch.save({"model_state": model.state_dict(), "val_metrics": va_m, "cfg": asdict(cfg)}, best_path)
            print(f"  ✓ Saved best (Dice={best_dice:.4f}) → {best_path}")


    unfreeze_all(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)


    for epoch in range(cfg.warmup_epochs_frozen+1, cfg.epochs+1):
        tr_loss, _ = run_epoch(train_loader, True)
        va_loss, va_m = run_epoch(val_loader, False)
        sch.step(va_loss)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_dice"].append(va_m["dice"])
        print(f"[{tag}] Epoch {epoch}/{cfg.epochs} | train={tr_loss:.4f} | val={va_loss:.4f} "
              f"| Acc={va_m['accuracy']:.4f} P/R/F1={va_m['precision']:.4f}/{va_m['recall']:.4f}/{va_m['f1']:.4f} "
              f"| IoU={va_m['iou']:.4f} | Dice={va_m['dice']:.4f}")
        if va_m['dice'] > best_dice:
            best_dice = va_m['dice']
            torch.save({"model_state": model.state_dict(), "val_metrics": va_m, "cfg": asdict(cfg)}, best_path)
            print(f"  ✓ Saved best (Dice={best_dice:.4f}) → {best_path}")

    def plot_curve(xs, ys, title, xlabel, ylabel, savepath):
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(xs, ys); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True)
        plt.savefig(savepath, bbox_inches="tight"); plt.close()


    xs = hist["epoch"]


    plot_curve(xs, hist["train_loss"], f"{tag} — Train Loss", "Epoch", "Loss", os.path.join(cfg.out_dir, f"{cfg.run_name}_{tag}_train_loss.png"))
    plot_curve(xs, hist["val_loss"],   f"{tag} — Val Loss",   "Epoch", "Loss", os.path.join(cfg.out_dir, f"{cfg.run_name}_{tag}_val_loss.png"))
    plot_curve(xs, hist["val_dice"],   f"{tag} — Val Dice",   "Epoch", "Dice", os.path.join(cfg.out_dir, f"{cfg.run_name}_{tag}_val_dice.png"))


    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tp=fp=fn=tn=total=0; test_loss=0.0

    
    with torch.no_grad():
        for imgs, msks, *_ in tqdm(test_loader, desc="Test", leave=False):
            imgs = imgs.to(device); msks = msks.to(device)
            logits = forward_logits(model, imgs)
            if logits.shape[1] != 1: logits = logits[:, :1, ...]
            test_loss += bce_dice_loss(logits, msks).item()
            a,b,c,d,t = seg_counts(logits, msks)
            tp+=a; fp+=b; fn+=c; tn+=d; total+=t

    test_loss /= len(test_loader)
    test_m = seg_metrics(tp,fp,fn,tn,total)
    return {"tag": tag, "best_val_dice": float(best_dice), "test_loss": float(test_loss), **test_m}
