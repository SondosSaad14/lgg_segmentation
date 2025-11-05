import os
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def show_predictions(model, loader, device, max_images: int=3, thr: float=0.5):
    model.eval()
    def logits_out(m, x):
        out = m(x)
        return out["out"] if isinstance(out, dict) and "out" in out else out
    shown = 0
    for imgs, msks, ips, mps, pids in loader:
        imgs = imgs.to(device)
        logits = logits_out(model, imgs)
        if logits.shape[1] != 1: logits = logits[:, :1, ...]
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > thr).float()
        for i in range(imgs.size(0)):
            if shown >= max_images: return
            img = imgs[i].cpu().permute(1,2,0).numpy()
            img = (img - img.min())/(img.max()-img.min()+1e-6)
            msk = msks[i].cpu().squeeze(0).numpy()
            prd = preds[i].squeeze(0).numpy()
            plt.figure(); plt.title(f"Input — {os.path.basename(ips[i])}"); plt.imshow(img); plt.axis("off"); plt.show()
            plt.figure(); plt.title("Ground Truth"); plt.imshow(msk, cmap="gray"); plt.axis("off"); plt.show()
            plt.figure(); plt.title("Prediction"); plt.imshow(prd, cmap="gray"); plt.axis("off"); plt.show()
            shown += 1
