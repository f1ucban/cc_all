import torch
from face.configs.base import cfg
from face.model.arcface_model import ArcFaceModel


def init_dm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArcFaceModel(cfg.n_cls).to(device)

    return device, model
