import cv2
import numpy as np
from facenet_pytorch import MTCNN
from web.utils.config import fd_thresh, min_eye_dist, min_face_sz
from web.face.f_prep import align_face, resize_face, preprocess_face
import time
from collections import deque
import torch
from web.utils.helpers import load_model
from web.utils.config import device, finetuned
import onnxruntime as ort
from pathlib import Path


class FaceProcessor:
    def __init__(self, device=None, model=None, quality_level="high", use_onnx=False):
        if device is None:
            self.device = device  # device from config
        else:
            self.device = device

        self.use_onnx = use_onnx
        self.mtcnn = MTCNN(
            min_face_size=40,
            keep_all=True,
            device=self.device,
            thresholds=[0.85, 0.9, 0.9],
        )

        if model is not None:
            if use_onnx:
                self.onnx_session = model
                self.model = None
            else:
                self.model = model
                self.onnx_session = None
        else:
            try:
                if use_onnx:
                    from web.utils.convert import create_onnx_session

                    onnx_path = Path(finetuned).parent / "arcface_model.onnx"
                    self.onnx_session = create_onnx_session(str(onnx_path))
                    self.model = None
                else:
                    self.model = load_model()
                    self.onnx_session = None
            except Exception as e:
                print(f"[FaceProcessor] Error loading model: {e}")
                self.model = None
                self.onnx_session = None

        if self.model is not None:
            self.model.eval()

        self.last_timestamp = 0
        self.frame_interval = 1 / 5
        self.fps_window = deque(maxlen=30)

        # Initialize configuration parameters
        self.min_eye_dist = min_eye_dist
        self.min_face_sz = min_face_sz
        self.debug = True

        # Quality settings with GPU batch sizes
        self.quality_levels = {
            "high": {
                "min_face_size": 100,
                "detection_confidence": 0.9,
                "batch_size": 32,  # GPU batch processing
                "gpu_memory_fraction": 0.8,
            },
            "medium": {
                "min_face_size": 80,
                "detection_confidence": 0.8,
                "batch_size": 16,
                "gpu_memory_fraction": 0.6,
            },
            "low": {
                "min_face_size": 60,
                "detection_confidence": 0.7,
                "batch_size": 8,
                "gpu_memory_fraction": 0.4,
            },
        }

        self.current_quality = quality_level
        self.update_quality_settings()

        print(
            f"[FaceProcessor] Initialized with quality: {quality_level} on device: {self.device} using {'ONNX' if use_onnx else 'PyTorch'}"
        )

    def update_quality_settings(self):
        settings = self.quality_levels[self.current_quality]
        self.min_face_size = settings["min_face_size"]
        self.detection_confidence = settings["detection_confidence"]
        self.batch_size = settings["batch_size"]
        self.gpu_memory_fraction = settings["gpu_memory_fraction"]

        # Update GPU memory fraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            print(
                f"[FaceProcessor] Updated GPU memory limit to {self.gpu_memory_fraction * 100}%"
            )

        print(f"[FaceProcessor] Updated quality settings to {self.current_quality}")

    def sync_frame(self):
        """Synchronize frame processing to maintain consistent timing"""
        current_time = time.time()
        elapsed = current_time - self.last_timestamp

        # Update FPS calculation
        if self.last_timestamp > 0:
            self.fps_window.append(1.0 / elapsed)
            avg_fps = sum(self.fps_window) / len(self.fps_window)
            print(f"[FaceProcessor] Current FPS: {avg_fps:.1f}")

        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)

        self.last_timestamp = time.time()

    def unk_face_result(self, bbox):
        return {
            "bbox": bbox,
            "identity": "unknown",
            "confidence": 0.0,
            "role": "",
        }

    def extract_embeds(self, img, boxes, probs, landmarks):
        embeds, bboxes, imgs = [], [], []
        print("[FaceProcessor]", f"\nProcessing {len(boxes)} detected faces")

        for i, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
            print("[FaceProcessor]", f"\nProcessing face {i+1}:")
            print("[FaceProcessor]", f"Detection confidence: {prob:.3f}")

            if prob < fd_thresh:
                print(
                    "[FaceProcessor]",
                    f"Face {i+1} rejected: confidence {prob:.3f} < threshold {fd_thresh}",
                )
                continue

            x1, y1, x2, y2 = map(int, box)
            face_sz = (x2 - x1) * (y2 - y1)
            print("[FaceProcessor]", f"Face size: {face_sz} pixels")

            try:
                print("[FaceProcessor]", "Aligning face...")
                aligned_face = align_face(img, box, lm)
                print("[FaceProcessor]", "Resizing face...")
                resized_face = resize_face(aligned_face)
                print("[FaceProcessor]", "Converting to OpenCV format...")
                face_cv = np.array(resized_face)[:, :, ::-1].copy()
                print("[FaceProcessor]", "Preprocessing face...")
                face_tensor = preprocess_face(face_cv, self.device)

                print("[FaceProcessor]", "Extracting embedding...")
                if self.use_onnx:
                    # Convert tensor to numpy for ONNX
                    face_np = face_tensor.cpu().numpy()
                    embed = self.onnx_session.run(None, {"input": face_np})[0]
                    embed = embed.flatten()
                else:
                    embed = self.model(face_tensor).detach().cpu().numpy().flatten()

                embeds.append(embed / np.linalg.norm(embed))
                bboxes.append([x1, y1, x2, y2])
                imgs.append(face_cv)
                print("[FaceProcessor]", f"Successfully processed face {i+1}")
            except Exception as e:
                print("[FaceProcessor]", f"Error processing face {i+1}: {str(e)}")
                continue

        return embeds, bboxes, imgs

    def process_face(self, img, require_single=True):
        print("[FaceProcessor]", "\nStarting face processing...")
        print("[FaceProcessor]", f"Input image shape: {img.shape}")

        # Synchronize frame processing
        self.sync_frame()

        print("[FaceProcessor]", "Running MTCNN detection...")
        boxes, _, lm = self.mtcnn.detect(img, landmarks=True)

        if boxes is None:
            print("[FaceProcessor]", "No faces detected")
            return None

        print("[FaceProcessor]", f"Detected {len(boxes)} faces")
        if require_single and len(boxes) != 1:
            print(
                "[FaceProcessor]",
                f"Rejected: {len(boxes)} faces detected, but require_single=True",
            )
            return None

        x1, y1, x2, y2 = map(int, boxes[0])
        face_sz = (x2 - x1) * (y2 - y1)
        print("[FaceProcessor]", f"Face size: {face_sz} pixels")

        # if face_sz < self.min_face_sz:
        #     print(
        #         "[FaceProcessor]",
        #         f"Face rejected: size {face_sz} < minimum {self.min_face_sz}",
        #     )
        #     return None

        eye_dist = np.linalg.norm(lm[0][0] - lm[0][1])
        print("[FaceProcessor]", f"Eye distance: {eye_dist:.2f} pixels")

        if eye_dist < self.min_eye_dist:
            print(
                "[FaceProcessor]",
                f"Face rejected: eye distance {eye_dist:.2f} < minimum {self.min_eye_dist}",
            )
            return None

        try:
            print("[FaceProcessor]", "Aligning face...")
            aligned_face = align_face(img, boxes[0], lm[0])
            print("[FaceProcessor]", "Resizing face...")
            resized_face = resize_face(aligned_face)
            print("[FaceProcessor]", "Converting to OpenCV format...")
            face_cv = np.array(resized_face)[:, :, ::-1].copy()
            print("[FaceProcessor]", "Preprocessing face...")
            face_tensor = preprocess_face(face_cv, self.device)
            print("[FaceProcessor]", "Extracting embedding...")

            if self.use_onnx:
                # Convert tensor to numpy for ONNX
                face_np = face_tensor.cpu().numpy()
                embed = self.onnx_session.run(None, {"input": face_np})[0]
                embed = embed.flatten()
            else:
                embed = self.model(face_tensor).detach().cpu().numpy().flatten()

            print("[FaceProcessor]", "Face processing completed successfully")
            return embed / np.linalg.norm(embed), face_cv
        except Exception as e:
            print("[FaceProcessor]", f"Error processing face: {str(e)}")
            return None
