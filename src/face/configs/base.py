from pathlib import Path
from easydict import EasyDict as edict


cfg = edict()


cfg.root = Path("../cc_all/dataset/processed/face")
cfg.pt_pth = Path("../cc_all/models/backbone/ms1mv3_arcface_r50_fp16.pth")
cfg.ckpt = Path("../cc_all/arcface_finetuned.pth")

cfg.n_cls = 60
cfg.embed_sz = 512
cfg.scale = 64.0
cfg.margin = 0.5
cfg.subc = 4

cfg.mean = [0.4674757421016693, 0.34854403138160706, 0.29534435272216797]
cfg.std = [0.32999464869499207, 0.2630784511566162, 0.23652763664722443]

cfg.bs = 64
cfg.nw = 8
cfg.os_factor = 5

cfg.lr = 6.5e-5
cfg.lr_layer4 = 5e-5
cfg.lr_arcloss = 1.5e-3
cfg.wd = 9e-6

cfg.dropout = 0.33
cfg.lbl_smth = 0.07

cfg.epochs = 350
cfg.patience = 35


cfg.arcIres50 = Path("../cc_all/models/finetuned/arcface_ires50_9383.pth")
