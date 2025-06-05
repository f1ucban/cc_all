import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.face.configs.base import cfg


class ArcFaceLoss(nn.Module):
    def __init__(self, in_feats, n_cls, s=cfg.scale, m=cfg.margin, k=cfg.subc):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(n_cls * k, in_feats))
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="leaky_relu")

        self.k = k
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = 1e-7

    def forward(self, x, lbl):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.view(-1, self.k, cosine.size(-1) // self.k)
        cosine = torch.max(cosine, dim=1)[0]
        cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, lbl.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        return output * self.s
