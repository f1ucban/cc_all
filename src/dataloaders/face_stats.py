import torch
import albumentations as A
from face.configs.base import cfg
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from dataloaders.face_dataset import FaceDataset


transform = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])
dataset = FaceDataset(root=cfg.root, split="train", transform=transform)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

ch_sum = torch.zeros(3)
ch_sq_sum = torch.zeros(3)
num_pixels = 0

for batch, _ in loader:
    ch_sum += torch.sum(batch, dim=[0, 2, 3])
    ch_sq_sum += torch.sum(batch**2, dim=[0, 2, 3])
    num_pixels += batch.size(0) * batch.size(2) * batch.size(3)

mean = ch_sum / num_pixels
std = torch.sqrt((ch_sq_sum / num_pixels) - (mean**2))

# print(f"Mean: {mean.tolist()}")
# print(f"Std: {std.tolist()}")
