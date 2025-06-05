import torch
import random
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from facenet_pytorch import MTCNN
from collections import defaultdict
from utils.logger import setup_logger
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split


logger = setup_logger("__faceDetector__")


FRAMES, FACE = Path("dataset/processed/frames/"), Path("dataset/processed/face/")
FACE.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    min_face_size=50, keep_all=True, device=device, thresholds=[0.85, 0.9, 0.9]
)


def get_properties(cropped_face):
    gray = np.array(cropped_face.convert("L"))
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    return (
        gray.mean(),
        gray.std(),
        cv.Laplacian(gray, cv.CV_64F).var(),
        hist[:40].sum(),
    )


def is_valid_face(brightness, contrast, laplacian, dark_ratio):
    score = (brightness / 255) * 0.2 + (contrast / 50) * 0.3 + (laplacian / 2000) * 0.5
    return (
        score > 0.3
        and brightness > 40
        and contrast > 20
        and laplacian > 100
        and dark_ratio < 0.25
    )


def align_face(image, box, landmarks):
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    rot_mat = cv.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated_img = cv.warpAffine(
        np.array(image), rot_mat, image.size, flags=cv.INTER_CUBIC
    )

    box_pts = np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
    )
    ones = np.ones((4, 1))
    box_pts = np.hstack([box_pts, ones])
    rotated_box = rot_mat.dot(box_pts.T).T

    x_coords = rotated_box[:, 0]
    y_coords = rotated_box[:, 1]
    x1, y1, x2, y2 = (
        int(min(x_coords)),
        int(min(y_coords)),
        int(max(x_coords)),
        int(max(y_coords)),
    )

    aligned_face = Image.fromarray(rotated_img).crop((x1, y1, x2, y2))

    return aligned_face


def resize(cropped_face, target_size=(112, 112), padding_color=(0, 0, 0)):
    w, h = cropped_face.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cropped_face.resize((new_w, new_h), Image.LANCZOS)

    padded_face = Image.new("RGB", target_size, padding_color)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    padded_face.paste(resized_image, (paste_x, paste_y))

    return padded_face


faces = defaultdict(list)
logger.info("Face detection and cropping process started.")


def process(subject):
    fv = subject / "fv"
    frames = list(fv.rglob("*.jpg"))
    detected, valid = 0, 0

    for frm_path in tqdm(frames, desc=f"Processing: {subject.name}"):
        frame = Image.open(frm_path).convert("RGB")
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

        if boxes is not None and landmarks is not None:
            for box, prob, lm in zip(boxes, probs, landmarks):
                if prob < 0.85:
                    continue

                detected += 1
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                cropped_face = align_face(frame, box, lm)

                if is_valid_face(*get_properties(cropped_face)):
                    resized_face = resize(cropped_face)
                    faces[subject.name].append(resized_face)
                    valid += 1

    logger.info(
        f"Folder '{subject}' processed. Faces detected: {detected}, Valid faces: {valid}"
    )


subjects = [
    s for split in ["train", "val", "test"] for s in (FRAMES / split).glob("s*")
]

with ThreadPoolExecutor(max_workers=12) as executor:
    executor.map(process, subjects)


random.seed(42)
limit = min(len(faces[subject]) for subject in faces)


for subject, images in faces.items():
    random.shuffle(images)
    images = images[:limit]

    train, val = train_test_split(images, train_size=0.8, random_state=42)

    for split, faces_in_split in [("train", train), ("val", val)]:
        dest = FACE / split / subject
        dest.mkdir(parents=True, exist_ok=True)
        for idx, face in enumerate(faces_in_split, 1):
            face.save(dest / f"face_{idx:04d}.jpg")

    logger.info(f"{subject}: Split saved - train={len(train)}, val={len(val)}")


logger.info("Face detection, cropping, and splitting done.")
