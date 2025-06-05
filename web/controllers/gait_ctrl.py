import cv2 as cv
import traceback
import numpy as np


from datetime import datetime
from werkzeug.utils import secure_filename


from flask import request
from web.database import db
from web.utils.config import enrollment
from web.utils.helpers import (
    error_response,
    success_response,
    create_user_dirs,
    cleanup_dir,
    validate_user_input,
    get_user_data,
)


class GaitController:
    def __init__(self, gait_proc, pose_controller):
        self.gait_proc = gait_proc
        self.pose_controller = pose_controller
        self.gait_feats = {}
        self.sess_states = {}

    def enroll_gait(self):
        try:
            self.gait_feats.clear()
            for sid in self.sess_states:
                self.sess_states[sid]["mode"] = "enrollment"
                self.sess_states[sid]["last_enrollment"] = datetime.now()

            user_data = get_user_data(
                request.form.get("firstname", "").strip(),
                request.form.get("lastname", "").strip(),
                request.form.get("role", "").strip(),
            )

            is_valid, error_msg = validate_user_input(
                user_data["firstname"], user_data["lastname"]
            )
            if not is_valid:
                return error_response(error_msg, 400)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enrollment.mkdir(parents=True, exist_ok=True)
            user_path = enrollment / f"{user_data['firstname']}_{user_data['lastname']}"
            gait_path = user_path / f"gait_{timestamp}"

            print(f"Creating directories: {user_path} and {gait_path}")
            success, error = create_user_dirs(user_path, timestamp, "gait")
            if not success:
                return error_response(f"Failed to create directories: {error}")

            if "video" not in request.files:
                return error_response("No valid video file provided", 400)

            video_file = request.files["video"]
            if not video_file:
                return error_response("No valid video file provided", 400)

            video_file.seek(0)
            orig_filename = secure_filename(video_file.filename)
            video_extension = (
                orig_filename.rsplit(".", 1)[1].lower()
                if "." in orig_filename
                else "mp4"
            )
            video_filename = f"gait_{timestamp}.{video_extension}"
            video_path = gait_path / video_filename

            print(f"Saving video to: {video_path}")
            video_file.save(str(video_path))
            if not (video_path.exists() and video_path.stat().st_size > 0):
                cleanup_dir(gait_path)
                return error_response("Failed to save video file")

            user_idx = db.get_user_by_name(
                user_data["firstname"], user_data["lastname"]
            )
            if not user_idx:
                user_idx = db.insert_user(
                    user_data["firstname"], user_data["lastname"], user_data["role"]
                )
                if not user_idx:
                    cleanup_dir(gait_path)
                    return error_response("Failed to create user")
            else:
                if hasattr(user_idx, "keys"):
                    user_idx = user_idx["user_idx"]

            current_enrollments = db.get_user_gait_enrollments(user_idx)
            max_enrollments = 5
            if len(current_enrollments) >= max_enrollments:
                cleanup_dir(gait_path)
                return error_response(
                    f"Maximum number of enrollments ({max_enrollments}) reached",
                    400,
                    {
                        "current_enrollments": len(current_enrollments),
                        "max_enrollments": max_enrollments,
                    },
                )

            features_list = []
            frame_count = 0

            print(f"Opening video file: {video_path}")
            cap = cv.VideoCapture(str(video_path))
            if not cap.isOpened():
                cleanup_dir(gait_path)
                return error_response("Could not open video file for processing")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                processed_frame, results = self.pose_controller.pose_proc.process_frame(
                    frame
                )
                if (
                    results
                    and hasattr(results, "pose_landmarks")
                    and results.pose_landmarks
                ):
                    features = self.gait_proc.extract_gait_features(
                        results.pose_landmarks
                    )
                    if features:
                        features_list.append(features)

            cap.release()

            if not features_list:
                cleanup_dir(gait_path)
                return error_response(
                    "No valid gait features extracted. Please ensure the video shows a clear view of the person walking.",
                    400,
                )

            avg_features = {}
            if features_list:
                for feature_name in features_list[0].keys():
                    values = [f[feature_name] for f in features_list]
                    avg_features[feature_name] = float(np.mean(values))

            feature_values = [
                avg_features[name] for name in self.gait_proc.feature_names
            ]
            features_array = np.array(feature_values, dtype=np.float32)
            features_path = gait_path / "gait_features.npy"
            np.save(str(features_path), features_array)

            features_bytes = features_array.tobytes()
            enrollment_idx = db.insert_gait_enrollment(user_idx, features_bytes)
            if not enrollment_idx:
                cleanup_dir(gait_path)
                return error_response("Failed to save gait features")

            updated_enrollments = db.get_user_gait_enrollments(user_idx)
            enrollment_count = len(updated_enrollments)

            self.gait_feats.clear()
            for sid in self.sess_states:
                self.sess_states[sid]["mode"] = "matching"
                self.sess_states[sid]["last_enrollment"] = datetime.now()

            return success_response(
                {
                    "user": {"user_idx": user_idx, **user_data},
                    "gait": str(gait_path),
                    "frame_count": frame_count,
                    "features_count": len(features_list),
                    "enrollment_count": enrollment_count,
                    "max_enrollments": max_enrollments,
                }
            )
        except Exception as e:
            print(f"Error in enroll_gait: {str(e)}")
            traceback.print_exc()
            return error_response(
                f"Error during gait enrollment: {str(e)}", include_traceback=True
            )

    def clear_gait_features(self):
        self.gait_feats.clear()
        return success_response(message="Cleared accumulated gait features")
