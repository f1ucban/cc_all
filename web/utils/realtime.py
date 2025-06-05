import base64
import cv2 as cv
import numpy as np
from flask_socketio import emit
import time
from datetime import datetime

from web.utils.config import known, unknown


class RealtimeEvents:
    def __init__(self, face_proc, face_match):
        self.face_proc = face_proc
        self.face_match = face_match
        self.last_saved_known_time = {}
        self.last_saved_unknown_time = None
        self.save_interval = 5

    def register_handlers(self, socketio):
        @socketio.on("process_frame")
        def handle_process_frame(data):
            try:
                if "frame" not in data:
                    emit("recognition_error", {"message": "No frame data provided"})
                    return

                frame_data = data["frame"]
                if ";base64," in frame_data:
                    header, frame_data = frame_data.split(";base64,")

                nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
                frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

                if frame is None:
                    emit(
                        "recognition_error", {"message": "Could not decode frame image"}
                    )
                    return

                recognition_results = {
                    "face_matches": [],
                }

                boxes, probs, landmarks = self.face_proc.mtcnn.detect(
                    frame, landmarks=True
                )
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
                                        or (
                                            current_time
                                            - self.last_saved_known_time[user_id]
                                        )
                                        > self.save_interval
                                    ):
                                        timestamp = datetime.now().strftime(
                                            "%Y%m%d_%H%M%S"
                                        )
                                        known_path = (
                                            known
                                            / str(user_id)
                                            / f"face_{timestamp}_{i}.jpg"
                                        )
                                        known_path.parent.mkdir(
                                            parents=True, exist_ok=True
                                        )
                                        cv.imwrite(str(known_path), face_img_bgr)
                                        self.last_saved_known_time[user_id] = (
                                            current_time
                                        )

                _, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 50])
                recognition_results["processed_frame"] = base64.b64encode(
                    buffer
                ).decode("utf-8")

                emit("recognition_results", recognition_results)

            except Exception as e:
                print(f"Error in process_frame SocketIO handler: {e}")
                emit("recognition_error", {"message": f"Server error: {e}"})
