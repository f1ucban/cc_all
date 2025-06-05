import io
import base64
import cv2 as cv
import numpy as np
from PIL import Image
from datetime import datetime
import time


from flask import request
from web.database import db
from web.face.f_prep import validate_face, combine_face
from web.utils.config import enrollment, min_req_face, known, unknown
from web.utils.helpers import (
    error_response,
    success_response,
    create_user_dirs,
    cleanup_dir,
    validate_user_input,
    get_user_data,
)


class FaceController:
    def __init__(self, face_proc, face_match):
        self.face_proc = face_proc
        self.face_match = face_match
        self.last_saved_known_time = {}
        self.last_saved_unknown_time = None
        self.save_interval = 5  

    def process_frame(self):
        if "frame" not in request.files:
            return error_response("No frame provided", 400)

        file = request.files["frame"]
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

        recognition_results = {
            "face_matches": [],
        }

        boxes, probs, landmarks = self.face_proc.mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            embeds, bboxes, imgs = self.face_proc.extract_embeds(
                frame, boxes, probs, landmarks
            )
            if embeds:
                face_results = self.face_match.match_faces(embeds, bboxes, imgs)
                recognition_results["face_matches"] = face_results

                current_time = time.time()

                for i, (face_img, result) in enumerate(zip(imgs, face_results)):
                    face_img_bgr = cv.cvtColor(face_img, cv.COLOR_RGB2BGR)

                    if result.get("identity") == "unknown":
                        if (
                            self.last_saved_unknown_time is None
                            or (current_time - self.last_saved_unknown_time)
                            > self.save_interval
                        ):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            unknown_path = unknown / f"face_{timestamp}_{i}.jpg"
                            cv.imwrite(str(unknown_path), face_img_bgr)
                            self.last_saved_unknown_time = current_time
                    else:
                        user_id = result.get("user_idx")
                        if user_id:
                            if (
                                user_id not in self.last_saved_known_time
                                or (current_time - self.last_saved_known_time[user_id])
                                > self.save_interval
                            ):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                known_path = (
                                    known / str(user_id) / f"face_{timestamp}_{i}.jpg"
                                )
                                known_path.parent.mkdir(parents=True, exist_ok=True)
                                cv.imwrite(str(known_path), face_img_bgr)
                                self.last_saved_known_time[user_id] = current_time

        _, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 50])
        recognition_results["processed_frame"] = base64.b64encode(buffer).decode(
            "utf-8"
        )

        return success_response(recognition_results)

    def enroll_face(self):
        if "firstname" not in request.form or "lastname" not in request.form:
            return error_response("Missing name parameters", 400)

        user_data = get_user_data(
            request.form["firstname"],
            request.form["lastname"],
            request.form.get("role", ""),
        )

        is_valid, error_msg = validate_user_input(
            user_data["firstname"], user_data["lastname"]
        )
        if not is_valid:
            return error_response(error_msg, 400)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_path = enrollment / f"{user_data['firstname']}_{user_data['lastname']}"
        face_path = user_path / f"face_{timestamp}"

        success, error = create_user_dirs(user_path, timestamp)
        if not success:
            return error_response(error)

        if "images" not in request.files:
            return error_response("No valid face images provided", 400)

        files = request.files.getlist("images")
        if not files:
            return error_response("No valid face images provided", 400)

        embeds = []
        saved_images = []
        successful_detections = 0
        failed_detections = 0

        for idx, file in enumerate(files):
            try:
                img = Image.open(io.BytesIO(file.read())).convert("RGB")
                img_np = np.array(img)

                result = self.face_proc.process_face(img_np)
                if result is not None:
                    embed, face_img = result
                    if face_img is not None and validate_face(embed):
                        embeds.append(embed)
                        img_path = face_path / f"face_{idx+1}.jpg"
                        try:
                            cv.imwrite(str(img_path), face_img)
                            saved_images.append(img_path)
                            successful_detections += 1
                        except Exception:
                            failed_detections += 1
                    else:
                        failed_detections += 1
                else:
                    failed_detections += 1
            except Exception:
                failed_detections += 1
                continue

        if len(embeds) < min_req_face:
            for img_path in saved_images:
                cleanup_dir(img_path)
            return error_response(
                f"Need at least {min_req_face} good quality face images. Got {len(embeds)} successful detections and {failed_detections} failed detections.",
                400,
            )

        user_idx = db.get_user_by_name(user_data["firstname"], user_data["lastname"])
        if not user_idx:
            user_idx = db.insert_user(
                user_data["firstname"], user_data["lastname"], user_data["role"]
            )
            if not user_idx:
                return error_response("Failed to create user")
        else:
            if hasattr(user_idx, "keys"):
                user_idx = user_idx["user_idx"]

        final_embedding = combine_face(embeds)
        if final_embedding is None:
            return error_response("Failed to combine face embeddings")

        pfp = face_path / "profile.jpg"
        try:
            cv.imwrite(str(pfp), face_img)
        except Exception:
            return error_response("Failed to save profile picture")

        features_path = face_path / "face_features.npy"
        np.save(str(features_path), final_embedding)
        db.insert_face_data(user_idx, final_embedding.tobytes())

        return success_response(
            {
                "user": {"user_idx": user_idx, **user_data},
                "face": str(face_path),
                "successful_detections": successful_detections,
                "failed_detections": failed_detections,
            }
        )
