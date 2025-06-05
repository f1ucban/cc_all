import traceback
import cv2 as cv
import mediapipe as mp
from web.gait.g_proc import GaitProcessor
import time
from collections import deque
from web.gait.const import MEDIAPIPE_CONFIG, FRAME_PARAMS


class PoseProcessor:
    def __init__(self, max_sequence_length=30):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(**MEDIAPIPE_CONFIG)
        self.gait_processor = GaitProcessor(max_sequence_length)

        self.last_frame_time = 0
        self.frame_interval = FRAME_PARAMS["frame_interval"]
        self.frame_times = deque(maxlen=FRAME_PARAMS["frame_buffer_size"])
        self.debug = True

        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=FRAME_PARAMS["drawing_specs"]["landmark"]["color"],
            thickness=FRAME_PARAMS["drawing_specs"]["landmark"]["thickness"],
            circle_radius=FRAME_PARAMS["drawing_specs"]["landmark"]["circle_radius"],
        )
        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=FRAME_PARAMS["drawing_specs"]["connection"]["color"],
            thickness=FRAME_PARAMS["drawing_specs"]["connection"]["thickness"],
            circle_radius=FRAME_PARAMS["drawing_specs"]["connection"]["circle_radius"],
        )

    def process_frame(self, frame):
        try:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = self.holistic.process(rgb_frame)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    self.landmark_spec,
                    self.connection_spec,
                )

            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                fps = 1.0 / (current_time - self.last_frame_time)
                self.frame_times.append(fps)
                avg_fps = sum(self.frame_times) / len(self.frame_times)
                if self.debug:
                    print(f"[PoseProcessor] Current FPS: {avg_fps:.1f}")

            self.last_frame_time = current_time

            return frame, results

        except Exception as e:
            if self.debug:
                print(f"[PoseProcessor] Error processing frame: {str(e)}")
                traceback.print_exc()
            return frame, None

    def get_gait_features(self):
        features = self.gait_processor.get_gait_features()
        if features:
            if self.debug:
                print(f"[PoseProcessor] Retrieved {len(features)} gait features")
        else:
            if self.debug:
                print("[PoseProcessor] No gait features available")
        return features

    def compare_gait_features(self, features1, features2):
        similarity = self.gait_processor.compare_gait_features(features1, features2)
        if self.debug:
            print(f"[PoseProcessor] Feature comparison similarity: {similarity:.3f}")
        return similarity

    def __del__(self):
        if self.debug:
            print("[PoseProcessor] Cleaning up resources")
        self.holistic.close()
