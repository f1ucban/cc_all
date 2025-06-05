import torch
import torch.nn as nn
import torch.nn.functional as F
from src.face.configs.base import cfg
from src.face.model.arcface_loss import ArcFaceLoss
from models.backbone.insightface.recognition.arcface_torch.backbones import iresnet50


class ArcFaceModel(nn.Module):
    def __init__(self, n_cls, pt_wghts=cfg.pt_pth):
        super().__init__()

        self.backbone = iresnet50(pretrained=False)
        self.backbone.load_state_dict(
            {k: v for k, v in torch.load(pt_wghts).items() if not k.startswith("fc.")},
            strict=False,
        )

        for name, param in self.backbone.named_parameters():
            param.requires_grad = True if "layer4" in name else False

        self.dropout = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_sz, cfg.embed_sz * 2),
            nn.PReLU(),
            nn.BatchNorm1d(cfg.embed_sz * 2),
            nn.Linear(cfg.embed_sz * 2, cfg.embed_sz),
            nn.PReLU(),
            nn.LayerNorm(cfg.embed_sz),
        )

        self.bn_feats = nn.BatchNorm1d(cfg.embed_sz)
        nn.init.constant_(self.bn_feats.weight, 1.0)
        nn.init.constant_(self.bn_feats.bias, 0.0)

        self.arc_loss = ArcFaceLoss(
            cfg.embed_sz, n_cls, s=cfg.scale, m=cfg.margin, k=cfg.subc
        )

    def forward(self, x, lbl=None):
        feats = self.bn_feats(self.dropout(self.backbone(x)))
        embeds = F.normalize(feats, p=2, dim=1)

        return (self.arc_loss(embeds, lbl), embeds) if lbl is not None else embeds
