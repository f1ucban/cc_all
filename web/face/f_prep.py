import torch
import cv2 as cv
import numpy as np
from PIL import Image
from web.utils.config import embed_sz


def validate_face(embed):
    if embed is None or not isinstance(embed, np.ndarray):
        return False

    if embed.shape[0] != embed_sz:
        return False

    if np.all(embed == 0):
        return False

    norm = np.linalg.norm(embed)
    if norm < 0.1 or norm > 10.0:
        return False

    return True


def standardize_face(embed):
    if embed is None:
        return None

    norm_embed = embed / np.linalg.norm(embed)
    mean = np.mean(norm_embed)
    std = np.std(norm_embed)

    if std < 1e-6:
        return norm_embed

    return (norm_embed - mean) / std


def combine_face(embeds):
    if not embeds:
        return None

    std_embeds = [standardize_face(embed) for embed in embeds]
    std_embeds = [e for e in std_embeds if e is not None]

    if not std_embeds:
        return None

    mean_embed = np.mean(std_embeds, axis=0)
    final_embed = mean_embed / np.linalg.norm(mean_embed)

    return final_embed.astype(np.float32)


def align_face(image, box, landmarks):
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    rot_mat = cv.getRotationMatrix2D(eye_center, angle, 1.0)

    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    dsize = (img_np.shape[1], img_np.shape[0])
    rotated_img = cv.warpAffine(img_np, rot_mat, dsize, flags=cv.INTER_CUBIC)
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


def resize_face(cropped_face, target_size=(112, 112), padding_color=(0, 0, 0)):
    w, h = cropped_face.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cropped_face.resize((new_w, new_h), Image.LANCZOS)

    padded_face = Image.new("RGB", target_size, padding_color)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    padded_face.paste(resized_image, (paste_x, paste_y))

    return padded_face


def preprocess_face(face_cv, device):
    face_rgb = cv.cvtColor(face_cv, cv.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_resized = resize_face(face_pil)
    face_tensor = torch.from_numpy(np.array(face_resized)).float()
    face_tensor = face_tensor / 255.0
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    return face_tensor
