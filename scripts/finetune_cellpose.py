import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models, train
from tifffile import imread


DEFAULT_CONFIG = {
    "data_root": "DATA/eGFP-LaminB3",
    "image_suffix": "_n2v_3d.tif",
    "mask_suffix": "_anno.tif",
    "model_dir": "models",
    "model_name": "cellpose_n2v_3d",
    "pretrained_model": "cpsam",
    "epochs": 100,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "batch_size": 4
}


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


def find_pairs(data_root: Path, image_suffix: str, mask_suffix: str):
    masks = sorted(data_root.rglob(f"*{mask_suffix}"))
    pairs = []

    for mask_path in masks:
        stem = mask_path.name[:-len(mask_suffix)]
        img_path = mask_path.with_name(stem + image_suffix)
        if img_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"missing image for mask: {mask_path}")

    return pairs


def vol_to_slices(img_path: Path, mask_path: Path):
    img = np.asarray(imread(img_path), dtype=np.float32)
    mask = np.asarray(imread(mask_path))

    if img.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {img_path.name}: image {img.shape}, mask {mask.shape}")

    imgs, masks = [], []
    for z in range(mask.shape[0]):
        if mask[z].max() > 0:
            imgs.append(img[z])
            masks.append(mask[z].astype(np.int32))

    print(f"{img_path.name}: {len(imgs)} annotated slices, {len(np.unique(mask)) - 1} objects")
    return imgs, masks


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose using *_n2v_3d.tif images and *_anno.tif masks.")
    parser.add_argument("--config", default=None,
                        help="Optional JSON config for data_root, suffixes, pretrained model, and training settings")
    parser.add_argument("--data_root", default=None,
                        help="Root directory to recursively scan for training pairs")
    parser.add_argument("--image_suffix", default=None,
                        help="Suffix for input image volumes")
    parser.add_argument("--mask_suffix", default=None,
                        help="Suffix for annotation mask volumes")
    parser.add_argument("--model_dir", default=None,
                        help="Directory where the fine-tuned model will be saved")
    parser.add_argument("--model_name", default=None,
                        help="Output model name")
    parser.add_argument("--pretrained_model", default=None,
                        help="Cellpose model name or path to an already fine-tuned model")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Training learning rate")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Training weight decay")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for Cellpose fine-tuning")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else dict(DEFAULT_CONFIG)

    data_root = Path(args.data_root or config["data_root"])
    image_suffix = args.image_suffix or config["image_suffix"]
    mask_suffix = args.mask_suffix or config["mask_suffix"]
    model_dir = Path(args.model_dir or config["model_dir"])
    model_name = args.model_name or config["model_name"]
    pretrained_model = args.pretrained_model or config["pretrained_model"]
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    learning_rate = args.learning_rate if args.learning_rate is not None else float(config["learning_rate"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config["weight_decay"])
    batch_size = args.batch_size if args.batch_size is not None else int(config["batch_size"])
    gpu = args.gpu or bool(config.get("gpu", False))

    pairs = find_pairs(data_root, image_suffix, mask_suffix)
    if not pairs:
        raise FileNotFoundError(
            f"No training pairs found under {data_root} with image suffix {image_suffix} and mask suffix {mask_suffix}"
        )

    train_imgs, train_masks = [], []
    for img_path, mask_path in pairs:
        imgs, masks = vol_to_slices(img_path, mask_path)
        train_imgs.extend(imgs)
        train_masks.extend(masks)

    if not train_imgs:
        raise ValueError("No annotated slices found in the paired training volumes.")

    print(f"\ntraining pairs : {len(pairs)}")
    print(f"training slices: {len(train_imgs)}")
    print("Starting Cellpose fine-tuning...")
    print(f"pretrained model: {pretrained_model}")
    print(f"epochs          : {epochs}")
    print(f"batch size      : {batch_size}")
    print(f"learning rate   : {learning_rate}")

    model = models.CellposeModel(gpu=args.gpu, pretrained_model=pretrained_model)

    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        normalize=True,
        n_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        min_train_masks=1,
        model_name=model_name,
        save_path=str(model_dir),
    )

    print(f"epochs logged   : {len(train_losses)}")
    print(f"final train loss: {train_losses[-1]:.4f}")
    print(f"best train loss : {np.min(train_losses):.4f}")

    if test_losses is not None and len(test_losses) > 0:
        print(f"final test loss : {test_losses[-1]:.4f}")
        print(f"best test loss  : {np.min(test_losses):.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    if test_losses is not None and len(test_losses) > 0:
        plt.plot(test_losses, label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()

    plot_path = model_dir / f"{model_name}_loss.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.show()

    print(f"\nModel saved to: {new_model_path}")
    print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
