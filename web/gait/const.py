import torch
from enum import Enum
import numpy as np


MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 2,
    "enable_segmentation": False,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
    "smooth_landmarks": True,
}


FRAME_PARAMS = {
    "target_fps": 30,
    "frame_interval": 1 / 30,
    "frame_buffer_size": 30,
    "drawing_specs": {
        "landmark": {
            "color": (245, 117, 66),
            "thickness": 2,
            "circle_radius": 4,
        },
        "connection": {
            "color": (245, 66, 230),
            "thickness": 2,
            "circle_radius": 2,
        },
    },
}


class QualityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


QUALITY_SETTINGS = {
    QualityLevel.HIGH: {
        "min_frames_for_cycle": 5,
        "frame_interval": 1 / 5,
        "visibility_threshold": 0.25,
        "min_visible_points": 3,
        "batch_size": 32,
    },
    QualityLevel.MEDIUM: {
        "min_frames_for_cycle": 4,
        "frame_interval": 1 / 4,
        "visibility_threshold": 0.2,
        "min_visible_points": 2,
        "batch_size": 16,
    },
    QualityLevel.LOW: {
        "min_frames_for_cycle": 3,
        "frame_interval": 1 / 3,
        "visibility_threshold": 0.15,
        "min_visible_points": 2,
        "batch_size": 8,
    },
}


DEVICE_CONFIG = {
    "default_device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "max_memory_usage": 0.8,
    "clear_cache_threshold": 0.9,
}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info():
    device = torch.device(DEVICE_CONFIG["default_device"])
    info = {
        "device": str(device),
        "is_cuda": device.type == "cuda",
        "memory_allocated": 0,
        "memory_cached": 0,
    }

    if device.type == "cuda":
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**2
        info["memory_cached"] = torch.cuda.memory_reserved() / 1024**2

    return info


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
