# MorphoNuc3D

Microscopy scripts for:
- Noise2Void denoising
- Cellpose fine-tuning
- 3D U-Net training
- point-cloud export

## Main Folders

- `scripts/`: training and prediction scripts
- `configs/`: JSON config files
- `notebooks/`: exploration and visualization

## File Convention

- raw input: `.czi`
- denoised image: `*_n2v_3d.tif`
- annotation mask: `*_anno.tif`

## Noise2Void

Train:

```bash
python scripts/train_n2v.py --config configs/id479_n2v.json
```

Predict:

```bash
python scripts/predict_n2v.py --config configs/id479_n2v.json
```

## Cellpose

Fine-tune from `*_n2v_3d.tif` and `*_anno.tif`:

```bash
python scripts/finetune_cellpose.py --config configs/finetune_cellpose.json
```

## 3D U-Net

Train:

```bash
python scripts/unet.py --config configs/unet.json --gpu
```

Predict:

- `notebooks/Unet_predict.ipynb`

## Point Cloud

```bash
python scripts/pointcloud.py --config configs/pointcloud.json
```

## Notes
- for new experiments, add a new config JSON
