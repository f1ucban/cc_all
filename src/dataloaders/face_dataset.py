import torch
import random
import cv2 as cv
import numpy as np
from pathlib import Path
from face.configs.base import cfg
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root=cfg.root, split="train", transform=None):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        idx = 0
        for subject in sorted(
            d for d in self.root.iterdir() if d.is_dir() and d.name != "unknown"
        ):
            self.class_to_idx[subject.name] = idx
            faces = list((self.root / subject.name).glob("*.jpg"))

            repeat = max(1, 100 // len(faces)) if split == "train" else 1
            if subject.name in ["s001", "s002"] and split == "train":
                repeat *= min(5, cfg.os_factor)

            self.samples.extend([(face, idx) for face in faces] * repeat)
            idx += 1

        if split == "val":
            unknowns = self.root / "unknown"
            if unknowns.exists():
                u_faces = list(
                    f
                    for u_subj in unknowns.iterdir()
                    if u_subj.is_dir()
                    for f in u_subj.glob("*.jpg")
                )
                n_known, n_unknown = len(self.samples), len(u_faces)
                max_unk = min(n_known // 3, n_unknown)
                self.samples.extend(
                    [(face, -1) for face in random.sample(u_faces, max_unk)]
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, lbl = self.samples[idx]
        face = cv.cvtColor(cv.imread(str(img)), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.transform:
            face = self.transform(image=face)["image"]

        return face, torch.tensor(lbl, dtype=torch.long)
