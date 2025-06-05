from face.configs.base import cfg
from torch.utils.data import DataLoader
from dataloaders.face_dataset import FaceDataset


import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2


def transform(split):
    base = [A.Normalize(mean=cfg.mean, std=cfg.std), ToTensorV2()]

    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.Affine(
                            translate_percent=(-0.1, 0.1),
                            scale=(0.9, 1.1),
                            rotate=(-7, 7),
                            shear=(-3, 3),
                            p=0.7,
                            border_mode=cv.BORDER_CONSTANT,
                        ),
                        A.Perspective(
                            scale=(0.02, 0.08),
                            keep_size=True,
                            fit_output=True,
                            interpolation=cv.INTER_LINEAR,
                            mask_interpolation=cv.INTER_NEAREST,
                            border_mode=cv.BORDER_CONSTANT,
                            p=0.3,
                        ),
                        A.ElasticTransform(
                            alpha=80,
                            sigma=18,
                            interpolation=cv.INTER_LINEAR,
                            mask_interpolation=cv.INTER_NEAREST,
                            approximate=False,
                            same_dxdy=True,
                            noise_distribution="gaussian",
                            p=0.3,
                        ),
                    ]
                ),
                A.ColorJitter(
                    brightness=(0.85, 1.15),
                    contrast=(0.85, 1.15),
                    saturation=(0.85, 1.15),
                    hue=(-0.025, 0.025),
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.CoarseDropout(
                            num_holes_range=(1, 2),
                            hole_height_range=(0.1, 0.2),
                            hole_width_range=(0.1, 0.2),
                            p=0.5,
                        ),
                        A.Erasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), p=0.6),
                    ],
                    p=0.4,
                ),
                A.Resize(
                    height=112,
                    width=112,
                    interpolation=cv.INTER_LINEAR,
                    mask_interpolation=cv.INTER_NEAREST,
                ),
                *base,
            ]
        )
    else:
        return A.Compose(base)


def dataloaders(root=cfg.root, batch_sz=cfg.bs, n_workers=cfg.nw):
    return {
        split: DataLoader(
            FaceDataset(root=root, split=split, transform=transform(split)),
            batch_size=batch_sz,
            shuffle=(split == "train"),
            num_workers=n_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=(n_workers > 0),
        )
        for split in ["train", "val"]
    }
