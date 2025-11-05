import os, json, argparse
import matplotlib.pyplot as plt
import torch

from .config import Config
from .data import make_dataloaders
from .models import (build_deeplabv3_resnet50,build_deeplabv3_resnet101,build_fcn_resnet50,build_lraspp_mobilenet_v3,ResUNet,)
from .train import train_eval_model


def main():
    parser = argparse.ArgumentParser(description="Train multiple segmentation models on LGG dataset.")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/lgg-mri-segmentation/kaggle_3m")
    parser.add_argument("--out_dir", type=str, default="artifacts_lgg_en")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.10)
    parser.add_argument("--test_split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        img_size=tuple(args.img_size),
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,).prepare()
    train_loader, val_loader, test_loader = make_dataloaders(cfg)
    loaders = (train_loader, val_loader, test_loader)



    builders = [("deeplabv3_resnet50",  build_deeplabv3_resnet50),("deeplabv3_resnet101", build_deeplabv3_resnet101),("fcn_resnet50",        build_fcn_resnet50),("lraspp_mobilenetv3",  build_lraspp_mobilenet_v3),("resunet",             lambda: ResUNet(base=64)),]

    results = []
    for tag, builder in builders:
        print(f"\n===== training {tag} =====")
        model = builder()
        res = train_eval_model(model, loaders, cfg, tag)
        results.append(res)
    with open(os.path.join(cfg.out_dir, f"{cfg.run_name}_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    best = max(results, key=lambda r: r["dice"])




    print("\n==== final comparison (test) ====")
    for r in results: print(r)
    print("\n==== recommended best model (by dice) ====")
    print(best)
    labels = [r["tag"] for r in results]
    scores = [r["dice"] for r in results]
    plt.figure()
    plt.bar(range(len(labels)), scores)
    plt.xticks(range(len(labels)), labels, rotation=15)
    plt.ylabel("dice")
    plt.title("M odel Comparison — Dice (Test)")
    plt.grid(True, axis="y")
    os.makedirs(cfg.out_dir, exist_ok=True)
    plt.savefig(os.path.join(cfg.out_dir, f"{cfg.run_name}_dice_comparison.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
