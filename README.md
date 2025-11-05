# LGG MRI Segmentation (PyTorch)

This package restructures your notebook into clean Python modules and CLI scripts for training and evaluating multiple segmentation models (DeepLabV3-ResNet50/101, FCN-ResNet50, LRASPP-MobileNetV3, and a custom ResUNet) on the **LGG MRI Segmentation** dataset.

## Install
```bash
pip install -r requirements.txt
```

## Dataset
Set `--data_dir` to the folder that contains `kaggle_3m/` with image/mask files (the structure from `mateuszbuda/lgg-mri-segmentation`). Pairing is discovered automatically.

## Train all models & compare
```bash
python -m lgg_segmentation.run_all --data_dir /path/to/kaggle_3m --out_dir artifacts_lgg_en
```
This will:
- Split by patient into train/val/test.
- Train each model with a short head warmup, then full finetune.
- Save the best checkpoint per model: `artifacts_lgg_en/lgg_multi_seg_en_<MODEL>_best.pt`.
- Write a JSON summary: `artifacts_lgg_en/lgg_multi_seg_en_summary.json`.
- Save comparison plot `..._dice_comparison.png`.

## Visualize predictions from the best model
```bash
python scripts/train_best_and_show.py --data_dir /path/to/kaggle_3m --out_dir artifacts_lgg_en --max_images 3
```

## Example results (structure)
See `examples/sample_results.json` for the schema your summary JSON will follow. **Numbers provided there are placeholders** so you can see the expected format.

## Notes
- Mixed precision is enabled automatically when CUDA is available.
- LRASPP head is made binary by replacing both `high_classifier` and `low_classifier` last convolutions.
- If you change `img_size`, previous checkpoints may not load (different head sizes). Train again or keep the same size.
