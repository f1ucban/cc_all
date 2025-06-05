import cv2 as cv
import numpy as np
from flask import request, Response
from web.utils.helpers import error_response


class PoseController:
    def __init__(self, pose_proc):
        self.pose_proc = pose_proc

    def detect_pose(self):
        if "image" not in request.files:
            return error_response("No image provided", 400)

        file = request.files["image"]
        img_array = np.frombuffer(file.read(), np.uint8)
        frame = cv.imdecode(img_array, cv.IMREAD_COLOR)
        processed_frame, _ = self.pose_proc.process_frame(frame)
        _, buffer = cv.imencode(".jpg", processed_frame)

        return Response(buffer.tobytes(), mimetype="image/jpeg")
