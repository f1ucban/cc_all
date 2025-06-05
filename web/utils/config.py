import torch
from pathlib import Path


class Config:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self._setup_paths()
        self._setup_thresholds()
        self._setup_parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_paths(self):
        self.database = self.root / "database" / "ccall.db"
        self.finetuned = self.root / "models" / "finetuned" / "arcface_ires50_9383.pth"
        self.uploads = self.root / "uploads"
        self.known = self.uploads / "known"
        self.unknown = self.uploads / "unknown"
        self.enrollment = self.uploads / "enrollment"
        self._create_directories()

    def _setup_thresholds(self):
        self.fd_thresh = 0.90
        self.uk_thresh = 0.95
        self.confi_thresh = 0.99
        self.min_face_sz = 3000
        self.min_eye_dist = 20
        self.fq_thresh = 0.85
        self.fsz_thresh = 40

    def _setup_parameters(self):
        self.min_req_face = 3
        self.embed_sz = 512
        self.accum_frms = 5
        self.cooldown = 5
        self.fps_win_sz = 30

    def _create_directories(self):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        for directory in [self.uploads, self.known, self.unknown, self.enrollment]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()

database = config.database
finetuned = config.finetuned
uploads = config.uploads
known = config.known
unknown = config.unknown
enrollment = config.enrollment
device = config.device
fd_thresh = config.fd_thresh
uk_thresh = config.uk_thresh
confi_thresh = config.confi_thresh
min_face_sz = config.min_face_sz
min_eye_dist = config.min_eye_dist
min_req_face = config.min_req_face
fq_thresh = config.fq_thresh
fsz_thresh = config.fsz_thresh
embed_sz = config.embed_sz
accum_frms = config.accum_frms
cooldown = config.cooldown
fps_win_sz = config.fps_win_sz
