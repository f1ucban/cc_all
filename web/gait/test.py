import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import deque, defaultdict

from datetime import datetime
import random
import time

from web import init_app
from web.database import db
from web.gait.pose import PoseProcessor
from web.gait.g_proc import GaitProcessor
from web.gait.g_match import GaitMatcher

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GaitTester:
    def __init__(self, video_path, output_dir=None):
        self.video_path = video_path
        self.pose_processor = PoseProcessor()
        self.gait_processor = GaitProcessor()
        self.gait_matcher = GaitMatcher(self.pose_processor, None)
        self.frame_count = 0
        self.feature_history = defaultdict(list)

        self.quality_levels = self.gait_processor.quality_levels
        self.current_quality = "high"
        self.min_frames = self.quality_levels[self.current_quality][
            "min_frames_for_analysis"
        ]
        self.max_frames = 300
        self.min_cycles = 3
        self.feature_weights = self.gait_processor._get_feature_weights()
        self.expected_features = self.gait_processor.feature_names

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.cwd() / "gait_test_results"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir.absolute()}")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enrolled_features = None
        self.matching_results = []

        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":0"

        app, _, _, _, _ = init_app()
        self.app = app

    def _validate_features(self, features):
        return self.gait_processor._validate_enhanced_features(features)

    def _calculate_feature_stability(self, features):
        return self.gait_processor._calculate_feature_stability(features)

    def process_video(self, test_matching=False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps
        last_frame_time = time.time()

        print(f"Processing video: {self.video_path}")
        print("Press Ctrl+C to stop processing")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Test matching mode: {test_matching}")

        try:
            while True:
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                processed_frame, results = self.pose_processor.process_frame(frame)

                features = None
                if (
                    results
                    and hasattr(results, "pose_landmarks")
                    and results.pose_landmarks
                ):
                    features = self.gait_processor.extract_gait_features(
                        results.pose_landmarks
                    )

                    if features and self._validate_features(features):
                        for feature_name, value in features.items():
                            self.feature_history[feature_name].append(value)

                        self._display_frame(processed_frame, features)

                        if test_matching:
                            self._test_matching(processed_frame, features)
                    else:
                        self._display_frame(
                            processed_frame, None, "Invalid or missing features"
                        )

                last_frame_time = current_time

        except KeyboardInterrupt:
            print("\nProcessing stopped by user")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            cap.release()

            if (
                test_matching
                and hasattr(self, "collected_features")
                and self.collected_features
            ):
                print("\nPerforming final matching with collected features...")
                try:
                    with self.app.app_context():
                        match_result = self.gait_matcher.match_gait(
                            self.collected_features
                        )
                        if match_result:
                            last_frame = self._display_frame_buffer.copy()

                            cv2.rectangle(
                                last_frame,
                                (0, 0),
                                (
                                    last_frame.shape[1],
                                    100,
                                ),
                                (0, 0, 0),
                                -1,
                            )

                            overlay = last_frame.copy()
                            cv2.rectangle(
                                overlay,
                                (0, 0),
                                (last_frame.shape[1], 200),
                                (0, 0, 0),
                                -1,
                            )
                            cv2.addWeighted(
                                overlay, 0.7, last_frame, 0.3, 0, last_frame
                            )

                            y_offset = 50
                            cv2.putText(
                                last_frame,
                                "Match Results:",
                                (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )
                            y_offset += 40

                            name = f"{match_result['firstname']} {match_result['lastname']}"
                            cv2.putText(
                                last_frame,
                                f"Name: {name}",
                                (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                            y_offset += 40

                            if match_result.get("role"):
                                cv2.putText(
                                    last_frame,
                                    f"Role: {match_result['role']}",
                                    (20, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                )
                                y_offset += 40

                            cv2.putText(
                                last_frame,
                                f"Confidence: {match_result['confidence']:.2%}",
                                (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                            cv2.imshow("Gait Analysis", last_frame)
                            cv2.waitKey(5000)  

                            results_file = (
                                self.output_dir
                                / f"matching_results_{self.timestamp}.txt"
                            )
                            with open(results_file, "w") as f:
                                f.write("Final Matching Results:\n")
                                f.write("=====================\n\n")
                                f.write(f"User: {name}\n")
                                if match_result.get("role"):
                                    f.write(f"Role: {match_result['role']}\n")
                                f.write(
                                    f"Confidence: {match_result['confidence']:.2%}\n"
                                )
                            print(
                                f"Matching results saved to: {results_file.absolute()}"
                            )
                except Exception as e:
                    print(f"Error during final matching: {str(e)}")
                    traceback.print_exc()

            if any(self.feature_history.values()):
                print("\nGenerating analysis plots...")
                self._generate_analysis_plots()
            else:
                print("No feature history to plot")

            cv2.destroyAllWindows()

    def enroll_gait(self, firstname, lastname, role=""):
        print(f"\nEnrolling gait for {firstname} {lastname}...")
        print("Press 'q' to quit, 's' to save enrollment features")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False

        features_list = []
        self.frame_count = 0 
        min_frames_for_enrollment = 30
        last_frame_time = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps

        try:
            while True:
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1  
                processed_frame, results = self.pose_processor.process_frame(frame)

                if (
                    results
                    and hasattr(results, "pose_landmarks")
                    and results.pose_landmarks
                ):
                    features = self.gait_processor.extract_gait_features(
                        results.pose_landmarks
                    )

                    if features:
                        features_list.append(features)
                        status = f"Enrolling... ({len(features_list)} frames collected)"
                        self._display_frame(processed_frame, features, status)
                    else:
                        self._display_frame(
                            processed_frame, None, "No features detected"
                        )
                else:
                    self._display_frame(processed_frame, None, "No pose detected")

                last_frame_time = current_time

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    if len(features_list) >= min_frames_for_enrollment:
                        print(
                            f"\nCollected {len(features_list)} frames of gait features"
                        )
                        self._save_enrollment_features(
                            features_list, firstname, lastname
                        )
                        break
                    else:
                        print(
                            f"Need at least {min_frames_for_enrollment} frames. Current: {len(features_list)}"
                        )

        except KeyboardInterrupt:
            print("\nEnrollment stopped by user")
        except Exception as e:
            print(f"Error during enrollment: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()

            if not features_list:
                print("No gait features were collected during enrollment")
                return False

            if len(features_list) < min_frames_for_enrollment:
                print(
                    f"Not enough frames collected. Need {min_frames_for_enrollment}, got {len(features_list)}"
                )
                return False

            avg_features = {}
            for feature_name in features_list[0].keys():
                values = [f[feature_name] for f in features_list]
                avg_features[feature_name] = float(np.mean(values))
                std_values = float(np.std(values))
                print(
                    f"{feature_name}: mean={avg_features[feature_name]:.3f}, std={std_values:.3f}"
                )

            feature_values = [
                avg_features[name] for name in self.gait_matcher.feature_names
            ]
            features_bytes = np.array(feature_values, dtype=np.float32).tobytes()
            print(f"Features bytes length: {len(features_bytes)}")

            try:
                with self.app.app_context():
                    print("Checking database for existing user before insertion...")
                    users_before = db.get_all_users()
                    print(f"Found {len(users_before)} users before check.")
                    user = next(
                        (
                            u
                            for u in users_before
                            if u["firstname"] == firstname and u["lastname"] == lastname
                        ),
                        None,
                    )

                    if user:
                        print(
                            f"User {firstname} {lastname} already exists with ID: {user['user_idx']}. Updating gait features..."
                        )
                        db.update_gait_features(user["user_idx"], features_bytes)
                        print(f"Called db.update_gait for user ID {user['user_idx']}")

                        print("Verifying gait features update...")
                        updated_users = db.get_all_users()
                        updated_user = next(
                            (
                                u
                                for u in updated_users
                                if u["user_idx"] == user["user_idx"]
                            ),
                            None,
                        )
                        if (
                            updated_user
                            and "gait_features" in updated_user
                            and updated_user["gait_features"]
                        ):
                            print(
                                "Successfully verified gait features were saved after update."
                            )
                            print(
                                f"Saved features length: {len(updated_user['gait_features'])} bytes"
                            )
                        else:
                            print(
                                "Warning: Could not verify gait features were saved after update."
                            )
                            print(
                                f"Updated user data: {dict(updated_user) if updated_user else None}"
                            )

                    else:
                        print(
                            f"User {firstname} {lastname} does not exist. Creating new user..."
                        )
                        insert_user_result = db.insert_user(firstname, lastname, role)
                        user_idx = insert_user_result 
                        print(f"Called db.insert_user, new user ID: {user_idx}")

                        db.insert_gait_enrollment(user_idx, features_bytes)
                        print(
                            f"Called db.insert_gait_enrollment for user ID {user_idx}"
                        )

                        print("Verifying gait features insertion...")
                        users_after_insert = db.get_all_users()
                        new_user = next(
                            (
                                u
                                for u in users_after_insert
                                if u["user_idx"] == user_idx
                            ),
                            None,
                        )
                        if (
                            new_user
                            and "gait_features" in new_user
                            and new_user["gait_features"]
                        ):
                            print(
                                "Successfully verified gait features were saved after insertion."
                            )
                            print(
                                f"Saved features length: {len(new_user['gait_features'])} bytes"
                            )
                        else:
                            print(
                                "Warning: Could not verify gait features were saved after insertion."
                            )
                            print(
                                f"New user data: {dict(new_user) if new_user else None}"
                            )

                self.enrolled_features = avg_features
                print("Enrollment process completed successfully.")
                return True
            except Exception as e:
                print(f"Error during enrollment: {str(e)}")
                import traceback

                traceback.print_exc()
                return False

    def _display_frame(self, frame, features, status=""):
        if not hasattr(self, "_display_frame_buffer"):
            self._display_frame_buffer = frame.copy()

        if frame is not None:
            self._display_frame_buffer = frame.copy()

        display_frame = self._display_frame_buffer
        y_offset = 30

        cv2.rectangle(
            display_frame, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1
        )

        if not hasattr(self, "_status_history"):
            self._status_history = deque(maxlen=30)  
            self._last_status_time = time.time()
            self._current_status = "Initializing..."
            self._status_count = 0

        current_time = time.time()

        if (
            current_time - self._last_status_time > 2.0  
            or (
                features and self._current_status != "Extracting features..."
            ) 
            or (
                not features and self._current_status != "Waiting for pose detection..."
            )
        ):  

            self._last_status_time = current_time
            if features:
                self._current_status = "Extracting features..."
                self._status_count += 1
            else:
                self._current_status = "Waiting for pose detection..."

            self._status_history.append(
                {
                    "status": self._current_status,
                    "time": current_time,
                    "frame": self.frame_count,
                }
            )

        cv2.putText(
            display_frame,
            self._current_status,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if features else (0, 0, 255),
            2,
        )
        y_offset += 25

        if hasattr(self, "_status_count") and self._status_count > 0:
            cv2.putText(
                display_frame,
                f"Feature extractions: {self._status_count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 25

        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(
            display_frame,
            frame_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if not hasattr(self, "_fps_history"):
            self._fps_history = deque(maxlen=30)

        if hasattr(self, "_last_frame_time"):
            fps = 1.0 / (current_time - self._last_frame_time)
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)

            cv2.putText(
                display_frame,
                f"FPS: {avg_fps:.1f}",
                (10, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        self._last_frame_time = current_time

        cv2.imshow("Gait Analysis", display_frame)
        cv2.waitKey(1) 

    def _test_matching(self, frame, current_features):
        if not current_features:
            return

        if not hasattr(self, "collected_features"):
            self.collected_features = []
        self.collected_features.append(current_features)

    def _save_current_features(self, features):
        if not features:
            print("No features to save")
            return

        output_file = self.output_dir / f"gait_features_{self.timestamp}.txt"
        with open(output_file, "w") as f:
            f.write(f"Frame: {self.frame_count}\n")
            for feature_name, value in features.items():
                f.write(f"{feature_name}: {value:.3f}\n")
        print(f"Features saved to {output_file}")

    def _save_enrollment_features(self, features_list, firstname, lastname):
        output_file = (
            self.output_dir / f"enrollment_{firstname}_{lastname}_{self.timestamp}.txt"
        )
        with open(output_file, "w") as f:
            f.write(f"Enrollment for {firstname} {lastname}\n")
            f.write(f"Total frames: {len(features_list)}\n\n")

            for i, features in enumerate(features_list):
                f.write(f"Frame {i+1}:\n")
                for feature_name, value in features.items():
                    f.write(f"{feature_name}: {value:.3f}\n")
                f.write("\n")
        print(f"Enrollment features saved to {output_file}")

    def _generate_analysis_plots(self):
        if not any(self.feature_history.values()):
            print("No features collected for analysis")
            return

        try:
            n_features = len(self.feature_history)
            fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
            fig.suptitle("Gait Feature Analysis Over Time")

            for idx, (feature_name, values) in enumerate(self.feature_history.items()):
                if values:
                    ax = axes[idx] if n_features > 1 else axes
                    ax.plot(values, label=feature_name)
                    ax.set_title(feature_name)
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Value")
                    ax.grid(True)
                    ax.legend()

            plt.tight_layout()
            plot_file = self.output_dir / f"gait_analysis_{self.timestamp}.png"
            plt.savefig(plot_file)
            plt.close()

            if plot_file.exists():
                print(f"Analysis plot saved to: {plot_file.absolute()}")
                print(f"Plot file size: {plot_file.stat().st_size} bytes")
            else:
                print("Error: Plot file was not created!")
        except Exception as e:
            print(f"Error generating analysis plots: {str(e)}")
            import traceback

            traceback.print_exc()

    def test_gait_matching(self):
        print("\nTesting gait matching with multiple enrollments...")

        test_landmarks_1 = self._create_test_landmarks()
        test_landmarks_2 = self._create_test_landmarks(variation=0.2)  # 20% variation
        test_landmarks_3 = self._create_test_landmarks(variation=0.5)  # 50% variation
        features_1 = self.gait_processor.extract_gait_features(test_landmarks_1)
        features_2 = self.gait_processor.extract_gait_features(test_landmarks_2)
        features_3 = self.gait_processor.extract_gait_features(test_landmarks_3)

        if not all([features_1, features_2, features_3]):
            print("Failed to extract features for all test cases")
            return

        features_1_norm = self.gait_processor._normalize_features(features_1)
        features_2_norm = self.gait_processor._normalize_features(features_2)
        features_3_norm = self.gait_processor._normalize_features(features_3)

        print("\nEnrolling users...")
        print("----------------------------------------")

        success_1 = self.gait_matcher.enroll_gait(1, features_1_norm)
        print(f"User 1 enrollment: {'Success' if success_1 else 'Failed'}")

        success_2 = self.gait_matcher.enroll_gait(2, features_2_norm)
        print(f"User 2 enrollment: {'Success' if success_2 else 'Failed'}")

        success_3 = self.gait_matcher.enroll_gait(3, features_3_norm)
        print(f"User 3 enrollment: {'Success' if success_3 else 'Failed'}")

        if not all([success_1, success_2, success_3]):
            print("Failed to enroll all users")
            return

        print("\nTesting matching...")
        print("----------------------------------------")

        test_cases = [
            ("Same user (User 1)", features_1_norm, 1),
            ("Similar user (User 2)", features_2_norm, 2),
            ("Different user (User 3)", features_3_norm, 3),
        ]

        for case_name, features, expected_user_id in test_cases:
            print(f"\nTest case: {case_name}")
            print("----------------------------------------")

            match_result = self.gait_matcher.match_gait(features)

            if match_result:
                print(
                    f"Matched user: {match_result['firstname']} {match_result['lastname']}"
                )
                print(f"Confidence: {match_result['confidence']:.4f}")

                quality_result = self.gait_matcher.verify_match_quality(
                    features, match_result.get("features", features)
                )
                print(f"Match quality: {'Passed' if quality_result else 'Failed'}")

                matched_user_id = match_result.get("user_id")
                if matched_user_id == expected_user_id:
                    print("✓ Correct user matched")
                else:
                    print(f"✗ Incorrect match - Expected user {expected_user_id}")
            else:
                print("No match found")

        print("\nTesting feature comparison...")
        print("----------------------------------------")

        for i, (name1, feat1) in enumerate(
            [
                ("User 1", features_1_norm),
                ("User 2", features_2_norm),
                ("User 3", features_3_norm),
            ]
        ):
            for j, (name2, feat2) in enumerate(
                [
                    ("User 1", features_1_norm),
                    ("User 2", features_2_norm),
                    ("User 3", features_3_norm),
                ]
            ):
                if i < j:  
                    print(f"\nComparing {name1} vs {name2}:")
                    print("----------------------------------------")

                    similarity = self.gait_matcher._calculate_weighted_similarity(
                        feat1, feat2
                    )
                    print(f"Similarity score: {similarity:.4f}")

                    quality = self.gait_matcher.verify_match_quality(feat1, feat2)
                    print(f"Match quality: {'Passed' if quality else 'Failed'}")

    def _create_test_landmarks(self, variation=0.0):
        num_frames = 30  
        landmarks = []

        base_params = {
            "arm_swing": {
                "amplitude": 0.3,
                "frequency": 1.0,
                "phase": 0.0,
                "asymmetry": 0.1,
            },
            "shoulder": {
                "roll": 0.1,
                "bounce": 0.05,
                "sway": 0.15,
                "width": 0.4,
            },
            "head": {
                "bob": 0.08,
                "sway": 0.1,
                "stability": 0.9,
            },
            "lower_body": {
                "knee_separation": 0.3,
                "stance_width": 0.35,
            },
            "coordination": {
                "smoothness": 0.85,
                "stability": 0.8,
                "efficiency": 0.75,
            },
        }

        for category in base_params:
            for param in base_params[category]:
                base_params[category][param] *= 1.0 + random.uniform(
                    -variation, variation
                )

        for frame in range(num_frames):
            t = frame / num_frames 

            phase = 2 * np.pi * t

            frame_landmarks = {
                # Left arm
                "left_shoulder": [
                    0.3 + base_params["shoulder"]["sway"] * np.sin(phase),
                    0.5 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                "left_elbow": [
                    0.3
                    + base_params["arm_swing"]["amplitude"]
                    * np.sin(phase + base_params["arm_swing"]["phase"]),
                    0.4 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                "left_wrist": [
                    0.3
                    + base_params["arm_swing"]["amplitude"]
                    * 1.5
                    * np.sin(phase + base_params["arm_swing"]["phase"]),
                    0.3 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                # Right arm
                "right_shoulder": [
                    0.7 + base_params["shoulder"]["sway"] * np.sin(phase + np.pi),
                    0.5 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                "right_elbow": [
                    0.7
                    + base_params["arm_swing"]["amplitude"]
                    * np.sin(phase + base_params["arm_swing"]["phase"] + np.pi),
                    0.4 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                "right_wrist": [
                    0.7
                    + base_params["arm_swing"]["amplitude"]
                    * 1.5
                    * np.sin(phase + base_params["arm_swing"]["phase"] + np.pi),
                    0.3 + base_params["shoulder"]["bounce"] * np.sin(2 * phase),
                    0.0,
                ],
                # Head and neck
                "nose": [
                    0.5 + base_params["head"]["sway"] * np.sin(phase),
                    0.8 + base_params["head"]["bob"] * np.sin(2 * phase),
                    0.0,
                ],
                "neck": [
                    0.5 + base_params["head"]["sway"] * 0.5 * np.sin(phase),
                    0.7 + base_params["head"]["bob"] * 0.5 * np.sin(2 * phase),
                    0.0,
                ],
                # Lower body
                "left_hip": [
                    0.4
                    + base_params["lower_body"]["stance_width"] * 0.5 * np.sin(phase),
                    0.3,
                    0.0,
                ],
                "right_hip": [
                    0.6
                    + base_params["lower_body"]["stance_width"]
                    * 0.5
                    * np.sin(phase + np.pi),
                    0.3,
                    0.0,
                ],
                "left_knee": [
                    0.4
                    + base_params["lower_body"]["knee_separation"]
                    * 0.5
                    * np.sin(phase),
                    0.2,
                    0.0,
                ],
                "right_knee": [
                    0.6
                    + base_params["lower_body"]["knee_separation"]
                    * 0.5
                    * np.sin(phase + np.pi),
                    0.2,
                    0.0,
                ],
            }

            for key in frame_landmarks:
                for i in range(3):
                    frame_landmarks[key][i] += random.uniform(-0.01, 0.01)

            landmarks.append(frame_landmarks)

        return landmarks


def main():
    parser = argparse.ArgumentParser(description="Test gait recognition system")
    parser.add_argument("video_path", help="Path to the test video file")
    parser.add_argument(
        "--output",
        help="Output directory for test results",
        default="gait_test_results",
    )
    parser.add_argument("--enroll", action="store_true", help="Run in enrollment mode")
    parser.add_argument("--firstname", help="First name for enrollment")
    parser.add_argument("--lastname", help="Last name for enrollment")
    parser.add_argument("--role", help="Role for enrollment", default="")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (match against enrolled users)",
    )
    args = parser.parse_args()

    tester = GaitTester(args.video_path, args.output)

    if args.enroll:
        if not args.firstname or not args.lastname:
            print("Error: First name and last name are required for enrollment")
            return
        tester.enroll_gait(args.firstname, args.lastname, args.role)
    else:
        # Always run in test mode if not enrolling
        tester.process_video(test_matching=True)


if __name__ == "__main__":
    main()
