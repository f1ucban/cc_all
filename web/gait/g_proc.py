import numpy as np
from collections import deque
from scipy.signal import welch
from scipy.stats import entropy
import time
import torch


class GaitProcessor:
    def __init__(self, max_sequence_length=60, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_sequence_length = max_sequence_length
        self.pose_sequence = deque(maxlen=max_sequence_length)

        self.max_signature_history = 50
        self.max_feature_history = 10
        self.memory_cleanup_threshold = 1000

        self.frame_counter = 0

        self.landmark_names = {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 7,
            "right_ear": 8,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }

        self.feature_names = [
            # UPPER BODY
            "arm_swing_asymmetry_mean",
            "arm_swing_asymmetry_std",
            "arm_swing_amplitude_left",
            "arm_swing_amplitude_right",
            "arm_swing_velocity_ratio",
            "arm_swing_frequency_left",
            "arm_swing_frequency_right",
            "arm_swing_phase_lag",
            "arm_swing_regularity_index",
            "elbow_bend_asymmetry",
            "wrist_trajectory_complexity",
            "upper_body_lean_angle",
            "shoulder_elbow_coordination",
            "upper_body_stability_index",
            "arm_swing_symmetry_index",
            "shoulder_height_variation",
            "upper_body_rhythm_consistency",
            # SHOULDER DYNAMICS
            "shoulder_roll_mean",
            "shoulder_roll_std",
            "shoulder_roll_asymmetry",
            "shoulder_width_variation_coeff",
            "shoulder_bounce_amplitude",
            "shoulder_bounce_frequency",
            "shoulder_sway_lateral_mean",
            "shoulder_sway_lateral_std",
            "shoulder_velocity_profile",
            "shoulder_acceleration_pattern",
            #  HEAD MOVEMENT PATTERNS
            "head_bob_amplitude",
            "head_bob_frequency",
            "head_sway_lateral_mean",
            "head_sway_pattern_regularity",
            "head_stability_index",
            "head_trajectory_smoothness",
            "head_shoulder_coordination",
            "neck_angle_variation",
            # SIMPLIFIED LOWER BODY
            "knee_separation_mean",
            "knee_separation_std",
            "knee_separation_rhythm",
            "stance_width_variation_coeff",
            # ADVANCED COORDINATION
            "movement_smoothness_global",
            "movement_smoothness_upper",
            "vertical_bounce_amplitude",
            "vertical_bounce_frequency",
            "movement_entropy_spatial",
            "movement_entropy_temporal",
            "coordination_index_arms_shoulders",
            "gait_stability_score",
            "movement_efficiency_index",
            "rhythmic_consistency_score",
            # UNIQUE INDIVIDUAL SIGNATURE
            "posture_signature",
            "movement_style_index",
            "energy_distribution_pattern",
            "bilateral_coordination_quality",
            "movement_complexity_index",
            "personal_rhythm_signature",
            "biomechanical_efficiency",
            "postural_sway_signature",
        ]

        self._initialize_buffers()
        self.quality_levels = {
            "high": {
                "min_frames_for_analysis": 5,
                "frame_interval": 1 / 5,
                "visibility_threshold": 0.5,
                "min_visible_points": 6,
                "temporal_smoothing": 0.7,
                "feature_stability_threshold": 0.7,
            },
            "medium": {
                "min_frames_for_analysis": 4,
                "frame_interval": 1 / 4,
                "visibility_threshold": 0.4,
                "min_visible_points": 5,
                "temporal_smoothing": 0.5,
                "feature_stability_threshold": 0.6,
            },
            "low": {
                "min_frames_for_analysis": 3,
                "frame_interval": 1 / 3,
                "visibility_threshold": 0.3,
                "min_visible_points": 4,
                "temporal_smoothing": 0.3,
                "feature_stability_threshold": 0.5,
            },
        }

        self.current_quality = "high"
        self.update_quality_settings()

        self.last_process_time = 0
        self.fps_window = deque(maxlen=30)
        self.prev_keypoints = None
        self.prev_time = None
        self.feature_history = deque(maxlen=10)

        self.signature_features = {}
        self.feature_stability_scores = {}

        print(
            f"[GaitProcessor] Initialized with {len(self.feature_names)} discriminative features"
        )

    def _initialize_buffers(self):

        self.arm_positions = deque(maxlen=self.max_sequence_length)
        self.shoulder_positions = deque(maxlen=self.max_sequence_length)
        self.head_positions = deque(maxlen=self.max_sequence_length)
        self.hip_positions = deque(maxlen=self.max_sequence_length)
        self.knee_positions = deque(maxlen=self.max_sequence_length)
        self.ankle_positions = deque(maxlen=self.max_sequence_length)
        self.body_center_positions = deque(maxlen=self.max_sequence_length)
        
        self.arm_swing_history = deque(maxlen=self.max_sequence_length)
        self.shoulder_dynamics = deque(maxlen=self.max_sequence_length)
        self.movement_quality = deque(maxlen=self.max_sequence_length)
        self.velocity_history = deque(maxlen=self.max_sequence_length)
        self.acceleration_history = deque(maxlen=self.max_sequence_length)

        self.temporal_patterns = deque(maxlen=self.max_sequence_length)
        self.coordination_patterns = deque(maxlen=self.max_sequence_length)

    def update_quality_settings(self):
        settings = self.quality_levels[self.current_quality]
        self.min_frames_for_analysis = settings["min_frames_for_analysis"]
        self.frame_interval = settings["frame_interval"]
        self.visibility_threshold = settings["visibility_threshold"]
        self.min_visible_points = settings["min_visible_points"]
        self.temporal_smoothing = settings["temporal_smoothing"]
        self.feature_stability_threshold = settings["feature_stability_threshold"]

    def should_process_frame(self):
        current_time = time.time()

        if self.last_process_time == 0:
            self.last_process_time = current_time
            return True

        elapsed = current_time - self.last_process_time

        if elapsed >= self.frame_interval:
            self.last_process_time = current_time
            return True

        return False

    def extract_gait_features(self, landmarks):
        try:
            if not self.should_process_frame():
                return None

            if not landmarks:
                return None

            keypoints = self._extract_keypoints(landmarks)
            if not keypoints:
                return None

            if not self._check_enhanced_visibility(keypoints):
                return None

            self._store_enhanced_frame_data(keypoints)

            if len(self.arm_positions) >= self.min_frames_for_analysis:
                features = self._calculate_enhanced_discriminative_features()
                if features and self._validate_enhanced_features(features):
                    self._update_signature_tracking(features)
                    return features

            return None

        except Exception as e:
            print(f"[GaitProcessor] Error extracting features: {str(e)}")
            return None

    def _extract_keypoints(self, landmarks):
        try:
            keypoints = {}

            for name, idx in self.landmark_names.items():
                if hasattr(landmarks, "landmark"):
                    lm = landmarks.landmark[idx]
                    keypoints[name] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": getattr(lm, "visibility", 1.0),
                    }
                else:
                    # Dictionary format
                    if name in landmarks:
                        keypoints[name] = landmarks[name]

            return keypoints

        except Exception as e:
            print(f"[GaitProcessor] Error extracting keypoints: {str(e)}")
            return None

    def _check_enhanced_visibility(self, keypoints):
        try:
            visible_count = 0
            visibility_quality = 0

            critical_points = [
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
            ]

            for point in critical_points:
                if point in keypoints:
                    visibility = keypoints[point].get("visibility", 1.0)
                    if visibility > self.visibility_threshold * 0.8:
                        visible_count += 1
                        visibility_quality += visibility

            avg_visibility = (
                visibility_quality / len(critical_points) if critical_points else 0
            )

            return (
                visible_count >= self.min_visible_points * 0.8
                or avg_visibility > self.visibility_threshold * 0.7
            )  

        except Exception as e:
            print(f"[GaitProcessor] Error checking visibility: {str(e)}")
            return True  

    def _store_enhanced_frame_data(self, keypoints):
        try:
            current_time = time.time()

            body_center = self._calculate_body_center(keypoints)
            self.body_center_positions.append(
                {"pos": body_center, "time": current_time}
            )

            arm_data = {
                "left_wrist": keypoints.get("left_wrist", {"x": 0, "y": 0}),
                "right_wrist": keypoints.get("right_wrist", {"x": 0, "y": 0}),
                "left_elbow": keypoints.get("left_elbow", {"x": 0, "y": 0}),
                "right_elbow": keypoints.get("right_elbow", {"x": 0, "y": 0}),
                "left_shoulder": keypoints.get("left_shoulder", {"x": 0, "y": 0}),
                "right_shoulder": keypoints.get("right_shoulder", {"x": 0, "y": 0}),
                "time": current_time,
            }

            arm_data["left_elbow_angle"] = self._calculate_arm_angle(
                arm_data["left_shoulder"],
                arm_data["left_elbow"],
                arm_data["left_wrist"],
            )
            arm_data["right_elbow_angle"] = self._calculate_arm_angle(
                arm_data["right_shoulder"],
                arm_data["right_elbow"],
                arm_data["right_wrist"],
            )

            self.arm_positions.append(arm_data)

            # Calculate velocities and accelerations if we have previous data
            if len(self.arm_positions) > 1:
                self._calculate_kinematics()

            self.shoulder_positions.append(
                {
                    "left": keypoints.get("left_shoulder", {"x": 0, "y": 0}),
                    "right": keypoints.get("right_shoulder", {"x": 0, "y": 0}),
                    "time": current_time,
                }
            )

            self.head_positions.append(
                {**keypoints.get("nose", {"x": 0, "y": 0}), "time": current_time}
            )

            self.hip_positions.append(
                {
                    "left": keypoints.get("left_hip", {"x": 0, "y": 0}),
                    "right": keypoints.get("right_hip", {"x": 0, "y": 0}),
                    "time": current_time,
                }
            )

            self.knee_positions.append(
                {
                    "left": keypoints.get("left_knee", {"x": 0, "y": 0}),
                    "right": keypoints.get("right_knee", {"x": 0, "y": 0}),
                    "time": current_time,
                }
            )

            self.ankle_positions.append(
                {
                    "left": keypoints.get("left_ankle", {"x": 0, "y": 0}),
                    "right": keypoints.get("right_ankle", {"x": 0, "y": 0}),
                    "time": current_time,
                }
            )

        except Exception as e:
            print(f"[GaitProcessor] Error storing frame data: {str(e)}")

    def _calculate_arm_angle(self, shoulder, elbow, wrist):
        """Calculate arm angle at elbow"""
        try:
            # Vector from elbow to shoulder
            v1 = np.array([shoulder["x"] - elbow["x"], shoulder["y"] - elbow["y"]])
            # Vector from elbow to wrist
            v2 = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"]])

            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)

            return float(angle)

        except:
            return 0.0

    def _calculate_kinematics(self):
        """Calculate velocity and acceleration from position history"""
        try:
            if len(self.arm_positions) < 2:
                return

            current = self.arm_positions[-1]
            previous = self.arm_positions[-2]

            dt = current["time"] - previous["time"]
            if dt <= 0:
                return

            # Calculate velocities for key points
            velocities = {}
            accelerations = {}

            for side in ["left", "right"]:
                for joint in ["wrist", "elbow", "shoulder"]:
                    key = f"{side}_{joint}"

                    # Velocity
                    dx = current[key]["x"] - previous[key]["x"]
                    dy = current[key]["y"] - previous[key]["y"]

                    vel_x = dx / dt
                    vel_y = dy / dt
                    vel_mag = np.sqrt(vel_x**2 + vel_y**2)

                    velocities[key] = {"x": vel_x, "y": vel_y, "magnitude": vel_mag}

            self.velocity_history.append(velocities)

            # Calculate accelerations if we have velocity history
            if len(self.velocity_history) >= 2:
                prev_vel = self.velocity_history[-2]

                for key in velocities:
                    if key in prev_vel:
                        acc_x = (velocities[key]["x"] - prev_vel[key]["x"]) / dt
                        acc_y = (velocities[key]["y"] - prev_vel[key]["y"]) / dt
                        acc_mag = np.sqrt(acc_x**2 + acc_y**2)

                        accelerations[key] = {
                            "x": acc_x,
                            "y": acc_y,
                            "magnitude": acc_mag,
                        }

                self.acceleration_history.append(accelerations)

        except Exception as e:
            print(f"[GaitProcessor] Error calculating kinematics: {str(e)}")

    def _calculate_body_center(self, keypoints):
        """Calculate body center from keypoints"""
        try:
            left_shoulder = keypoints.get("left_shoulder", {"x": 0, "y": 0})
            right_shoulder = keypoints.get("right_shoulder", {"x": 0, "y": 0})
            left_hip = keypoints.get("left_hip", {"x": 0, "y": 0})
            right_hip = keypoints.get("right_hip", {"x": 0, "y": 0})

            center_x = (
                left_shoulder["x"]
                + right_shoulder["x"]
                + left_hip["x"]
                + right_hip["x"]
            ) / 4
            center_y = (
                left_shoulder["y"]
                + right_shoulder["y"]
                + left_hip["y"]
                + right_hip["y"]
            ) / 4

            return {"x": center_x, "y": center_y}

        except Exception as e:
            return {"x": 0.5, "y": 0.5}

    def _calculate_enhanced_discriminative_features(self):
        try:
            features = {}

            # body size reference for normalization
            body_size = self._get_enhanced_body_size_reference()
            if body_size <= 0:
                return None

            # 1. arm swing features (primary discriminators)
            arm_features = self._calculate_enhanced_arm_features(body_size)
            features.update(arm_features)

            # 2. shoulder dynamics
            shoulder_features = self._calculate_enhanced_shoulder_features(body_size)
            features.update(shoulder_features)

            # 3. head movement patterns
            head_features = self._calculate_enhanced_head_features(body_size)
            features.update(head_features)

            # 4. hip and pelvic dynamics
            hip_features = self._calculate_enhanced_hip_features(body_size)
            features.update(hip_features)

            # 5. leg movement patterns
            leg_features = self._calculate_enhanced_leg_features(body_size)
            features.update(leg_features)

            # 6. Advanced temporal and coordination features
            coordination_features = self._calculate_advanced_coordination_features()
            features.update(coordination_features)

            # 7. Individual signature features
            signature_features = self._calculate_signature_features(body_size)
            features.update(signature_features)

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating enhanced features: {str(e)}")
            return None

    def _get_enhanced_body_size_reference(self):
        try:
            if not self.shoulder_positions or not self.hip_positions:
                return 0

            shoulder_widths = []
            torso_heights = []

            min_len = min(len(self.shoulder_positions), len(self.hip_positions))

            for i in range(min_len):
                shoulder_data = self.shoulder_positions[i]
                hip_data = self.hip_positions[i]

                # Shoulder width
                shoulder_width = abs(
                    shoulder_data["left"]["x"] - shoulder_data["right"]["x"]
                )
                shoulder_widths.append(shoulder_width)

                # Torso height (shoulder to hip midpoint)
                shoulder_center_y = (
                    shoulder_data["left"]["y"] + shoulder_data["right"]["y"]
                ) / 2
                hip_center_y = (hip_data["left"]["y"] + hip_data["right"]["y"]) / 2
                torso_height = abs(shoulder_center_y - hip_center_y)
                torso_heights.append(torso_height)

            if shoulder_widths and torso_heights:
                avg_width = np.mean(shoulder_widths)
                avg_height = np.mean(torso_heights)
                return (avg_width + avg_height) / 2
            elif shoulder_widths:
                return np.mean(shoulder_widths)
            else:
                return 0

        except Exception as e:
            return 0

    def _calculate_enhanced_arm_features(self, body_size):
        try:
            features = {}

            left_arm_movements = []
            right_arm_movements = []
            left_velocities = []
            right_velocities = []
            arm_asymmetries = []
            elbow_angles_left = []
            elbow_angles_right = []
            wrist_trajectories_left = []
            wrist_trajectories_right = []
            shoulder_heights = []
            upper_body_angles = []

            for i, arm_data in enumerate(self.arm_positions):
                if i < len(self.shoulder_positions):
                    shoulder_data = self.shoulder_positions[i]

                    left_swing = np.sqrt(
                        (arm_data["left_wrist"]["x"] - shoulder_data["left"]["x"]) ** 2
                        + (arm_data["left_wrist"]["y"] - shoulder_data["left"]["y"])
                        ** 2
                    )
                    right_swing = np.sqrt(
                        (arm_data["right_wrist"]["x"] - shoulder_data["right"]["x"])
                        ** 2
                        + (arm_data["right_wrist"]["y"] - shoulder_data["right"]["y"])
                        ** 2
                    )

                    left_arm_movements.append(left_swing)
                    right_arm_movements.append(right_swing)

                    # Asymmetry
                    asymmetry = abs(left_swing - right_swing)
                    arm_asymmetries.append(asymmetry)

                    # Elbow angles
                    elbow_angles_left.append(arm_data.get("left_elbow_angle", 0))
                    elbow_angles_right.append(arm_data.get("right_elbow_angle", 0))

                    # Wrist trajectory complexity (distance from straight line)
                    if len(self.arm_positions) > 2:
                        wrist_trajectories_left.append(arm_data["left_wrist"]["x"])
                        wrist_trajectories_right.append(arm_data["right_wrist"]["x"])

                    # Shoulder height variation
                    shoulder_height = (
                        shoulder_data["left"]["y"] + shoulder_data["right"]["y"]
                    ) / 2
                    shoulder_heights.append(shoulder_height)

                    # Upper body lean angle
                    shoulder_center_x = (
                        shoulder_data["left"]["x"] + shoulder_data["right"]["x"]
                    ) / 2
                    shoulder_center_y = (
                        shoulder_data["left"]["y"] + shoulder_data["right"]["y"]
                    ) / 2
                    if i < len(self.hip_positions):
                        hip_data = self.hip_positions[i]
                        hip_center_x = (
                            hip_data["left"]["x"] + hip_data["right"]["x"]
                        ) / 2
                        hip_center_y = (
                            hip_data["left"]["y"] + hip_data["right"]["y"]
                        ) / 2
                        lean_angle = np.arctan2(
                            hip_center_x - shoulder_center_x,
                            hip_center_y - shoulder_center_y,
                        )
                        upper_body_angles.append(lean_angle)

            # velocities if available
            if self.velocity_history:
                for vel_data in self.velocity_history:
                    if "left_wrist" in vel_data and "right_wrist" in vel_data:
                        left_velocities.append(vel_data["left_wrist"]["magnitude"])
                        right_velocities.append(vel_data["right_wrist"]["magnitude"])

            if left_arm_movements and right_arm_movements:
                # Normalize by body size
                left_arm_movements = np.array(left_arm_movements) / body_size
                right_arm_movements = np.array(right_arm_movements) / body_size
                arm_asymmetries = np.array(arm_asymmetries) / body_size

                # Enhanced asymmetry features
                features["arm_swing_asymmetry_mean"] = float(np.mean(arm_asymmetries))
                features["arm_swing_asymmetry_std"] = float(np.std(arm_asymmetries))

                # Individual arm characteristics
                features["arm_swing_amplitude_left"] = float(
                    np.mean(left_arm_movements)
                )
                features["arm_swing_amplitude_right"] = float(
                    np.mean(right_arm_movements)
                )

                # Velocity ratio (discriminative for individuals)
                if left_velocities and right_velocities:
                    left_vel_mean = np.mean(left_velocities)
                    right_vel_mean = np.mean(right_velocities)
                    features["arm_swing_velocity_ratio"] = float(
                        left_vel_mean / right_vel_mean if right_vel_mean > 0 else 1.0
                    )
                else:
                    features["arm_swing_velocity_ratio"] = 1.0

                # Individual arm frequencies
                if len(left_arm_movements) > 8:
                    # Left arm frequency
                    freqs_l, powers_l = welch(
                        left_arm_movements, nperseg=min(len(left_arm_movements), 8)
                    )
                    dominant_freq_idx_l = np.argmax(powers_l[1:]) + 1
                    features["arm_swing_frequency_left"] = float(
                        freqs_l[dominant_freq_idx_l]
                    )

                    # Right arm frequency
                    freqs_r, powers_r = welch(
                        right_arm_movements, nperseg=min(len(right_arm_movements), 8)
                    )
                    dominant_freq_idx_r = np.argmax(powers_r[1:]) + 1
                    features["arm_swing_frequency_right"] = float(
                        freqs_r[dominant_freq_idx_r]
                    )
                else:
                    features["arm_swing_frequency_left"] = 0.0
                    features["arm_swing_frequency_right"] = 0.0

                # Phase lag between arms (enhanced)
                if len(left_arm_movements) > 6:
                    correlation = np.correlate(
                        left_arm_movements, right_arm_movements, mode="full"
                    )
                    lag = np.argmax(correlation) - len(right_arm_movements) + 1
                    features["arm_swing_phase_lag"] = float(abs(lag))
                else:
                    features["arm_swing_phase_lag"] = 0.0

                # Regularity index (how consistent the arm swing is)
                left_regularity = (
                    1.0
                    / (1.0 + np.std(left_arm_movements) / np.mean(left_arm_movements))
                    if np.mean(left_arm_movements) > 0
                    else 0
                )
                right_regularity = (
                    1.0
                    / (1.0 + np.std(right_arm_movements) / np.mean(right_arm_movements))
                    if np.mean(right_arm_movements) > 0
                    else 0
                )
                features["arm_swing_regularity_index"] = float(
                    (left_regularity + right_regularity) / 2
                )

                # Elbow bend asymmetry
                if elbow_angles_left and elbow_angles_right:
                    elbow_diff = np.abs(
                        np.array(elbow_angles_left) - np.array(elbow_angles_right)
                    )
                    features["elbow_bend_asymmetry"] = float(np.mean(elbow_diff))
                else:
                    features["elbow_bend_asymmetry"] = 0.0

                # Wrist trajectory complexity
                if (
                    len(wrist_trajectories_left) > 3
                    and len(wrist_trajectories_right) > 3
                ):
                    # Calculate path complexity using smoothness measure
                    left_complexity = self._calculate_trajectory_complexity(
                        wrist_trajectories_left
                    )
                    right_complexity = self._calculate_trajectory_complexity(
                        wrist_trajectories_right
                    )
                    features["wrist_trajectory_complexity"] = float(
                        (left_complexity + right_complexity) / 2
                    )
                else:
                    features["wrist_trajectory_complexity"] = 0.0

                # New front-view specific features

                # Upper body lean angle
                if upper_body_angles:
                    features["upper_body_lean_angle"] = float(
                        np.mean(upper_body_angles)
                    )
                else:
                    features["upper_body_lean_angle"] = 0.0

                # Shoulder-elbow coordination
                if len(elbow_angles_left) > 2 and len(elbow_angles_right) > 2:
                    left_coord = self._calculate_coordination(
                        elbow_angles_left, left_arm_movements
                    )
                    right_coord = self._calculate_coordination(
                        elbow_angles_right, right_arm_movements
                    )
                    features["shoulder_elbow_coordination"] = float(
                        (left_coord + right_coord) / 2
                    )
                else:
                    features["shoulder_elbow_coordination"] = 0.0

                # Upper body stability index
                if shoulder_heights:
                    height_variation = np.std(shoulder_heights) / body_size
                    features["upper_body_stability_index"] = float(
                        1.0 / (1.0 + height_variation)
                    )
                else:
                    features["upper_body_stability_index"] = 1.0

                # Arm swing symmetry index
                if len(left_arm_movements) > 1 and len(right_arm_movements) > 1:
                    symmetry = 1.0 - np.mean(
                        np.abs(left_arm_movements - right_arm_movements)
                    ) / np.mean(left_arm_movements + right_arm_movements)
                    features["arm_swing_symmetry_index"] = float(symmetry)
                else:
                    features["arm_swing_symmetry_index"] = 0.0

                # Shoulder height variation
                if shoulder_heights:
                    features["shoulder_height_variation"] = float(
                        np.std(shoulder_heights) / body_size
                    )
                else:
                    features["shoulder_height_variation"] = 0.0

                # Upper body rhythm consistency
                if len(left_arm_movements) > 8 and len(right_arm_movements) > 8:
                    left_rhythm = self._calculate_rhythm_consistency(left_arm_movements)
                    right_rhythm = self._calculate_rhythm_consistency(
                        right_arm_movements
                    )
                    features["upper_body_rhythm_consistency"] = float(
                        (left_rhythm + right_rhythm) / 2
                    )
                else:
                    features["upper_body_rhythm_consistency"] = 0.0

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating arm features: {str(e)}")
            return {}

    def _calculate_coordination(self, angles, movements):
        """Calculate coordination between angles and movements"""
        try:
            if len(angles) < 2 or len(movements) < 2:
                return 0.0

            # Calculate correlation between angles and movements
            correlation = np.corrcoef(angles, movements)[0, 1]
            return float(abs(correlation) if not np.isnan(correlation) else 0.0)

        except:
            return 0.0

    def _calculate_rhythm_consistency(self, signal):
        """Calculate rhythm consistency using autocorrelation"""
        try:
            if len(signal) < 8:
                return 0.0

            signal = np.array(signal)

            # Normalize signal
            signal = (
                (signal - np.mean(signal)) / np.std(signal)
                if np.std(signal) > 0
                else signal
            )

            # Calculate autocorrelation
            autocorr = np.correlate(signal, signal, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Find peak autocorrelation (excluding lag 0)
            if len(autocorr) > 1:
                peak_corr = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
                return float(max(0, peak_corr))

            return 0.0

        except:
            return 0.0

    def _calculate_trajectory_complexity(self, trajectory):
        """Calculate trajectory complexity using smoothness measures"""
        try:
            if len(trajectory) < 4:
                return 0.0

            trajectory = np.array(trajectory)

            # Calculate second derivative (acceleration/jerk)
            first_diff = np.diff(trajectory)
            second_diff = np.diff(first_diff)

            # Complexity as variance in acceleration
            complexity = np.var(second_diff) if len(second_diff) > 0 else 0.0
            return float(complexity)

        except:
            return 0.0

    def _calculate_enhanced_shoulder_features(self, body_size):
        """Calculate enhanced shoulder dynamics features"""
        try:
            features = {}

            shoulder_rolls = []
            shoulder_widths = []
            shoulder_heights_left = []
            shoulder_heights_right = []
            shoulder_velocities = []
            shoulder_accelerations = []

            for i, shoulder_data in enumerate(self.shoulder_positions):
                # Shoulder roll (tilt)
                roll = shoulder_data["left"]["y"] - shoulder_data["right"]["y"]
                shoulder_rolls.append(roll)

                # Shoulder width
                width = abs(shoulder_data["left"]["x"] - shoulder_data["right"]["x"])
                shoulder_widths.append(width)

                # Individual shoulder heights
                shoulder_heights_left.append(shoulder_data["left"]["y"])
                shoulder_heights_right.append(shoulder_data["right"]["y"])

            # Get shoulder velocities from velocity history
            if self.velocity_history:
                for vel_data in self.velocity_history:
                    if "left_shoulder" in vel_data and "right_shoulder" in vel_data:
                        avg_vel = (
                            vel_data["left_shoulder"]["magnitude"]
                            + vel_data["right_shoulder"]["magnitude"]
                        ) / 2
                        shoulder_velocities.append(avg_vel)

            # Get shoulder accelerations
            if self.acceleration_history:
                for acc_data in self.acceleration_history:
                    if "left_shoulder" in acc_data and "right_shoulder" in acc_data:
                        avg_acc = (
                            acc_data["left_shoulder"]["magnitude"]
                            + acc_data["right_shoulder"]["magnitude"]
                        ) / 2
                        shoulder_accelerations.append(avg_acc)

            if shoulder_rolls and shoulder_widths:
                # Normalize by body size
                shoulder_rolls = np.array(shoulder_rolls) / body_size
                shoulder_widths = np.array(shoulder_widths) / body_size

                # Basic shoulder roll features
                features["shoulder_roll_mean"] = float(np.mean(shoulder_rolls))
                features["shoulder_roll_std"] = float(np.std(shoulder_rolls))

                # Shoulder asymmetry
                left_height_var = np.var(shoulder_heights_left)
                right_height_var = np.var(shoulder_heights_right)
                features["shoulder_roll_asymmetry"] = float(
                    abs(left_height_var - right_height_var)
                )

                # Shoulder width variation coefficient
                width_mean = np.mean(shoulder_widths)
                width_std = np.std(shoulder_widths)
                features["shoulder_width_variation_coeff"] = float(
                    width_std / width_mean if width_mean > 0 else 0
                )

                # Shoulder bounce (vertical movement)
                shoulder_centers_y = [
                    (shoulder_heights_left[i] + shoulder_heights_right[i]) / 2
                    for i in range(len(shoulder_heights_left))
                ]
                if len(shoulder_centers_y) > 1:
                    features["shoulder_bounce_amplitude"] = float(
                        np.std(shoulder_centers_y) / body_size
                    )

                    # Shoulder bounce frequency
                    if len(shoulder_centers_y) > 8:
                        freqs, powers = welch(
                            shoulder_centers_y, nperseg=min(len(shoulder_centers_y), 8)
                        )
                        if len(powers) > 1:
                            dominant_freq_idx = np.argmax(powers[1:]) + 1
                            features["shoulder_bounce_frequency"] = float(
                                freqs[dominant_freq_idx]
                            )
                        else:
                            features["shoulder_bounce_frequency"] = 0.0
                    else:
                        features["shoulder_bounce_frequency"] = 0.0
                else:
                    features["shoulder_bounce_amplitude"] = 0.0
                    features["shoulder_bounce_frequency"] = 0.0

                # Lateral sway
                shoulder_centers_x = [
                    (shoulder_data["left"]["x"] + shoulder_data["right"]["x"]) / 2
                    for shoulder_data in self.shoulder_positions
                ]
                if len(shoulder_centers_x) > 1:
                    features["shoulder_sway_lateral_mean"] = float(
                        np.mean(shoulder_centers_x)
                    )
                    features["shoulder_sway_lateral_std"] = float(
                        np.std(shoulder_centers_x) / body_size
                    )
                else:
                    features["shoulder_sway_lateral_mean"] = 0.5
                    features["shoulder_sway_lateral_std"] = 0.0

                # Velocity and acceleration profiles
                if shoulder_velocities:
                    features["shoulder_velocity_profile"] = float(
                        np.mean(shoulder_velocities)
                    )
                else:
                    features["shoulder_velocity_profile"] = 0.0

                if shoulder_accelerations:
                    features["shoulder_acceleration_pattern"] = float(
                        np.mean(shoulder_accelerations)
                    )
                else:
                    features["shoulder_acceleration_pattern"] = 0.0

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating shoulder features: {str(e)}")
            return {}

    def _calculate_enhanced_head_features(self, body_size):
        """Calculate enhanced head movement features"""
        try:
            features = {}

            head_positions_x = []
            head_positions_y = []
            head_shoulder_angles = []

            for i, head_data in enumerate(self.head_positions):
                head_positions_x.append(head_data["x"])
                head_positions_y.append(head_data["y"])

                # Head-shoulder coordination angle
                if i < len(self.shoulder_positions):
                    shoulder_data = self.shoulder_positions[i]
                    shoulder_center_x = (
                        shoulder_data["left"]["x"] + shoulder_data["right"]["x"]
                    ) / 2
                    shoulder_center_y = (
                        shoulder_data["left"]["y"] + shoulder_data["right"]["y"]
                    ) / 2

                    # Angle between head and shoulder center
                    dx = head_data["x"] - shoulder_center_x
                    dy = head_data["y"] - shoulder_center_y
                    angle = np.arctan2(dy, dx)
                    head_shoulder_angles.append(angle)

            if head_positions_x and head_positions_y:
                # Head bob (vertical movement)
                head_bob_amplitude = np.std(head_positions_y) / body_size
                features["head_bob_amplitude"] = float(head_bob_amplitude)

                # Head bob frequency
                if len(head_positions_y) > 8:
                    freqs, powers = welch(
                        head_positions_y, nperseg=min(len(head_positions_y), 8)
                    )
                    if len(powers) > 1:
                        dominant_freq_idx = np.argmax(powers[1:]) + 1
                        features["head_bob_frequency"] = float(freqs[dominant_freq_idx])
                    else:
                        features["head_bob_frequency"] = 0.0
                else:
                    features["head_bob_frequency"] = 0.0

                # Lateral head sway
                features["head_sway_lateral_mean"] = float(np.mean(head_positions_x))

                # Head sway pattern regularity
                if len(head_positions_x) > 4:
                    sway_regularity = (
                        1.0
                        / (1.0 + np.std(head_positions_x) / np.mean(head_positions_x))
                        if np.mean(head_positions_x) > 0
                        else 0
                    )
                    features["head_sway_pattern_regularity"] = float(sway_regularity)
                else:
                    features["head_sway_pattern_regularity"] = 0.0

                # Head stability index
                head_movement_total = np.sqrt(
                    np.var(head_positions_x) + np.var(head_positions_y)
                )
                features["head_stability_index"] = float(
                    1.0 / (1.0 + head_movement_total / body_size)
                )

                # Trajectory smoothness
                if len(head_positions_x) > 3:
                    x_complexity = self._calculate_trajectory_complexity(
                        head_positions_x
                    )
                    y_complexity = self._calculate_trajectory_complexity(
                        head_positions_y
                    )
                    features["head_trajectory_smoothness"] = float(
                        1.0 / (1.0 + x_complexity + y_complexity)
                    )
                else:
                    features["head_trajectory_smoothness"] = 1.0

                # Head-shoulder coordination
                if head_shoulder_angles:
                    features["head_shoulder_coordination"] = float(
                        1.0 / (1.0 + np.std(head_shoulder_angles))
                    )
                else:
                    features["head_shoulder_coordination"] = 1.0

                # Neck angle variation
                if len(head_shoulder_angles) > 1:
                    features["neck_angle_variation"] = float(
                        np.std(head_shoulder_angles)
                    )
                else:
                    features["neck_angle_variation"] = 0.0

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating head features: {str(e)}")
            return {}

    def _calculate_enhanced_hip_features(self, body_size):
        """Calculate enhanced hip and pelvic dynamics features"""
        try:
            features = {}

            hip_centers_x = []
            hip_centers_y = []
            hip_widths = []
            hip_heights_left = []
            hip_heights_right = []
            pelvic_tilts = []

            for hip_data in self.hip_positions:
                # Hip center
                center_x = (hip_data["left"]["x"] + hip_data["right"]["x"]) / 2
                center_y = (hip_data["left"]["y"] + hip_data["right"]["y"]) / 2
                hip_centers_x.append(center_x)
                hip_centers_y.append(center_y)

                # Hip width
                width = abs(hip_data["left"]["x"] - hip_data["right"]["x"])
                hip_widths.append(width)

                # Individual hip heights
                hip_heights_left.append(hip_data["left"]["y"])
                hip_heights_right.append(hip_data["right"]["y"])

                # Pelvic tilt
                tilt = hip_data["left"]["y"] - hip_data["right"]["y"]
                pelvic_tilts.append(tilt)

            if hip_centers_x and hip_widths:
                # Normalize by body size
                hip_centers_x = np.array(hip_centers_x)
                hip_widths = np.array(hip_widths) / body_size
                pelvic_tilts = np.array(pelvic_tilts) / body_size

                # Hip sway amplitude
                features["hip_sway_amplitude_mean"] = float(
                    np.std(hip_centers_x) / body_size
                )
                features["hip_sway_amplitude_std"] = float(
                    np.std(hip_centers_x) / body_size
                )

                # Hip sway frequency
                if len(hip_centers_x) > 8:
                    freqs, powers = welch(
                        hip_centers_x, nperseg=min(len(hip_centers_x), 8)
                    )
                    if len(powers) > 1:
                        dominant_freq_idx = np.argmax(powers[1:]) + 1
                        features["hip_sway_frequency"] = float(freqs[dominant_freq_idx])
                    else:
                        features["hip_sway_frequency"] = 0.0
                else:
                    features["hip_sway_frequency"] = 0.0

                # Hip sway asymmetry
                left_var = np.var(hip_heights_left)
                right_var = np.var(hip_heights_right)
                features["hip_sway_asymmetry"] = float(abs(left_var - right_var))

                # Hip width variation coefficient
                width_mean = np.mean(hip_widths)
                width_std = np.std(hip_widths)
                features["hip_width_variation_coeff"] = float(
                    width_std / width_mean if width_mean > 0 else 0
                )

                # Hip drop asymmetry
                features["hip_drop_asymmetry_mean"] = float(np.mean(pelvic_tilts))
                features["hip_drop_asymmetry_std"] = float(np.std(pelvic_tilts))

                # Pelvic rotation indicator (simplified)
                if len(hip_centers_x) > 2:
                    rotation_changes = np.diff(pelvic_tilts)
                    features["pelvic_rotation_indicator"] = float(
                        np.std(rotation_changes)
                    )
                else:
                    features["pelvic_rotation_indicator"] = 0.0

                # Hip velocity signature
                if self.velocity_history:
                    hip_velocities = []
                    for vel_data in self.velocity_history:
                        if "left_hip" in vel_data and "right_hip" in vel_data:
                            avg_vel = (
                                vel_data["left_hip"]["magnitude"]
                                + vel_data["right_hip"]["magnitude"]
                            ) / 2
                            hip_velocities.append(avg_vel)

                    if hip_velocities:
                        features["hip_velocity_signature"] = float(
                            np.mean(hip_velocities)
                        )
                    else:
                        features["hip_velocity_signature"] = 0.0
                else:
                    features["hip_velocity_signature"] = 0.0

                # Pelvic stability index
                hip_movement_total = np.sqrt(
                    np.var(hip_centers_x) + np.var(hip_centers_y)
                )
                features["pelvic_stability_index"] = float(
                    1.0 / (1.0 + hip_movement_total / body_size)
                )

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating hip features: {str(e)}")
            return {}

    def _calculate_enhanced_leg_features(self, body_size):
        """Calculate enhanced leg movement features"""
        try:
            features = {}

            knee_separations = []
            ankle_positions_x = []
            ankle_positions_y = []
            stance_widths = []

            for i, knee_data in enumerate(self.knee_positions):
                # Knee separation
                separation = abs(knee_data["left"]["x"] - knee_data["right"]["x"])
                knee_separations.append(separation)

                # Ankle positions for stance width
                if i < len(self.ankle_positions):
                    ankle_data = self.ankle_positions[i]
                    ankle_positions_x.extend(
                        [ankle_data["left"]["x"], ankle_data["right"]["x"]]
                    )
                    ankle_positions_y.extend(
                        [ankle_data["left"]["y"], ankle_data["right"]["y"]]
                    )

                    stance_width = abs(
                        ankle_data["left"]["x"] - ankle_data["right"]["x"]
                    )
                    stance_widths.append(stance_width)

            if knee_separations:
                # Normalize by body size
                knee_separations = np.array(knee_separations) / body_size
                stance_widths = (
                    np.array(stance_widths) / body_size
                    if stance_widths
                    else np.array([])
                )

                # Knee separation features
                features["knee_separation_mean"] = float(np.mean(knee_separations))
                features["knee_separation_std"] = float(np.std(knee_separations))

                # Knee separation rhythm
                if len(knee_separations) > 8:
                    freqs, powers = welch(
                        knee_separations, nperseg=min(len(knee_separations), 8)
                    )
                    if len(powers) > 1:
                        dominant_freq_idx = np.argmax(powers[1:]) + 1
                        features["knee_separation_rhythm"] = float(
                            freqs[dominant_freq_idx]
                        )
                    else:
                        features["knee_separation_rhythm"] = 0.0
                else:
                    features["knee_separation_rhythm"] = 0.0

                # Leg swing asymmetry
                if len(self.knee_positions) > 1:
                    left_knee_x = [pos["left"]["x"] for pos in self.knee_positions]
                    right_knee_x = [pos["right"]["x"] for pos in self.knee_positions]

                    left_var = np.var(left_knee_x)
                    right_var = np.var(right_knee_x)
                    features["leg_swing_asymmetry_lateral"] = float(
                        abs(left_var - right_var)
                    )
                else:
                    features["leg_swing_asymmetry_lateral"] = 0.0

                # Ankle sway features
                if ankle_positions_x:
                    features["ankle_sway_amplitude"] = float(
                        np.std(ankle_positions_x) / body_size
                    )

                    if len(ankle_positions_x) > 8:
                        freqs, powers = welch(
                            ankle_positions_x, nperseg=min(len(ankle_positions_x), 8)
                        )
                        if len(powers) > 1:
                            dominant_freq_idx = np.argmax(powers[1:]) + 1
                            features["ankle_sway_frequency"] = float(
                                freqs[dominant_freq_idx]
                            )
                        else:
                            features["ankle_sway_frequency"] = 0.0
                    else:
                        features["ankle_sway_frequency"] = 0.0
                else:
                    features["ankle_sway_amplitude"] = 0.0
                    features["ankle_sway_frequency"] = 0.0

                # Stance width variation
                if len(stance_widths) > 1:
                    stance_mean = np.mean(stance_widths)
                    stance_std = np.std(stance_widths)
                    features["stance_width_variation_coeff"] = float(
                        stance_std / stance_mean if stance_mean > 0 else 0
                    )
                else:
                    features["stance_width_variation_coeff"] = 0.0

                # Knee-ankle coordination
                if len(self.knee_positions) > 1 and len(self.ankle_positions) > 1:
                    knee_ankle_correlation = self._calculate_knee_ankle_coordination()
                    features["knee_ankle_coordination"] = float(knee_ankle_correlation)
                else:
                    features["knee_ankle_coordination"] = 0.0

                # Apparent leg length ratio (perspective effect)
                if len(self.hip_positions) > 0 and len(self.ankle_positions) > 0:
                    leg_length_ratio = self._calculate_apparent_leg_length_ratio(
                        body_size
                    )
                    features["leg_length_apparent_ratio"] = float(leg_length_ratio)
                else:
                    features["leg_length_apparent_ratio"] = 1.0

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating leg features: {str(e)}")
            return {}

    def _calculate_knee_ankle_coordination(self):
        """Calculate coordination between knee and ankle movements"""
        try:
            knee_movements = []
            ankle_movements = []

            min_len = min(len(self.knee_positions), len(self.ankle_positions))

            for i in range(min_len):
                knee_data = self.knee_positions[i]
                ankle_data = self.ankle_positions[i]

                # Calculate movement magnitude for knees and ankles
                knee_center_x = (knee_data["left"]["x"] + knee_data["right"]["x"]) / 2
                ankle_center_x = (
                    ankle_data["left"]["x"] + ankle_data["right"]["x"]
                ) / 2

                knee_movements.append(knee_center_x)
                ankle_movements.append(ankle_center_x)

            if len(knee_movements) > 2 and len(ankle_movements) > 2:
                # Calculate correlation
                correlation = np.corrcoef(knee_movements, ankle_movements)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0

            return 0.0

        except:
            return 0.0

    def _calculate_apparent_leg_length_ratio(self, body_size):
        """Calculate apparent leg length ratio (perspective effects)"""
        try:
            left_leg_lengths = []
            right_leg_lengths = []

            min_len = min(len(self.hip_positions), len(self.ankle_positions))

            for i in range(min_len):
                hip_data = self.hip_positions[i]
                ankle_data = self.ankle_positions[i]

                # Left leg apparent length
                left_length = np.sqrt(
                    (hip_data["left"]["x"] - ankle_data["left"]["x"]) ** 2
                    + (hip_data["left"]["y"] - ankle_data["left"]["y"]) ** 2
                )
                left_leg_lengths.append(left_length)

                # Right leg apparent length
                right_length = np.sqrt(
                    (hip_data["right"]["x"] - ankle_data["right"]["x"]) ** 2
                    + (hip_data["right"]["y"] - ankle_data["right"]["y"]) ** 2
                )
                right_leg_lengths.append(right_length)

            if left_leg_lengths and right_leg_lengths:
                left_mean = np.mean(left_leg_lengths)
                right_mean = np.mean(right_leg_lengths)

                return left_mean / right_mean if right_mean > 0 else 1.0

            return 1.0

        except:
            return 1.0

    def _calculate_advanced_coordination_features(self):
        """Calculate advanced temporal and coordination features"""
        try:
            features = {}

            # Global movement smoothness
            if len(self.body_center_positions) > 3:
                body_x = [pos["pos"]["x"] for pos in self.body_center_positions]
                body_y = [pos["pos"]["y"] for pos in self.body_center_positions]

                x_smoothness = self._calculate_movement_smoothness(body_x)
                y_smoothness = self._calculate_movement_smoothness(body_y)
                features["movement_smoothness_global"] = float(
                    (x_smoothness + y_smoothness) / 2
                )
            else:
                features["movement_smoothness_global"] = 1.0

            # Upper body smoothness (shoulders and arms)
            if len(self.shoulder_positions) > 3:
                shoulder_x = [
                    (pos["left"]["x"] + pos["right"]["x"]) / 2
                    for pos in self.shoulder_positions
                ]
                features["movement_smoothness_upper"] = float(
                    self._calculate_movement_smoothness(shoulder_x)
                )
            else:
                features["movement_smoothness_upper"] = 1.0

            # Lower body smoothness (hips)
            if len(self.hip_positions) > 3:
                hip_x = [
                    (pos["left"]["x"] + pos["right"]["x"]) / 2
                    for pos in self.hip_positions
                ]
                features["movement_smoothness_lower"] = float(
                    self._calculate_movement_smoothness(hip_x)
                )
            else:
                features["movement_smoothness_lower"] = 1.0

            # Vertical bounce
            if len(self.body_center_positions) > 1:
                body_y = [pos["pos"]["y"] for pos in self.body_center_positions]
                features["vertical_bounce_amplitude"] = float(np.std(body_y))

                if len(body_y) > 8:
                    freqs, powers = welch(body_y, nperseg=min(len(body_y), 8))
                    if len(powers) > 1:
                        dominant_freq_idx = np.argmax(powers[1:]) + 1
                        features["vertical_bounce_frequency"] = float(
                            freqs[dominant_freq_idx]
                        )
                    else:
                        features["vertical_bounce_frequency"] = 0.0
                else:
                    features["vertical_bounce_frequency"] = 0.0
            else:
                features["vertical_bounce_amplitude"] = 0.0
                features["vertical_bounce_frequency"] = 0.0

            # Movement entropy (complexity measures)
            features["movement_entropy_spatial"] = self._calculate_spatial_entropy()
            features["movement_entropy_temporal"] = self._calculate_temporal_entropy()

            # Coordination indices
            features["coordination_index_arms_shoulders"] = (
                self._calculate_arms_shoulders_coordination()
            )
            features["coordination_index_shoulders_hips"] = (
                self._calculate_shoulders_hips_coordination()
            )
            features["coordination_index_hips_legs"] = (
                self._calculate_hips_legs_coordination()
            )

            # Overall stability and efficiency
            features["gait_stability_score"] = self._calculate_gait_stability()
            features["movement_efficiency_index"] = (
                self._calculate_movement_efficiency()
            )
            features["rhythmic_consistency_score"] = (
                self._calculate_rhythmic_consistency()
            )

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating coordination features: {str(e)}")
            return {}

    def _calculate_movement_smoothness(self, trajectory):
        """Calculate movement smoothness using jerk analysis"""
        try:
            if len(trajectory) < 4:
                return 1.0

            trajectory = np.array(trajectory)

            # Calculate jerk (third derivative)
            vel = np.diff(trajectory)
            acc = np.diff(vel)
            jerk = np.diff(acc)

            # Smoothness as inverse of jerk variance
            jerk_var = np.var(jerk) if len(jerk) > 0 else 0
            smoothness = 1.0 / (1.0 + jerk_var)

            return smoothness

        except:
            return 1.0

    def _calculate_spatial_entropy(self):
        """Calculate spatial movement entropy"""
        try:
            if len(self.body_center_positions) < 8:
                return 0.0

            positions_x = [pos["pos"]["x"] for pos in self.body_center_positions]
            positions_y = [pos["pos"]["y"] for pos in self.body_center_positions]

            # Discretize positions into bins
            hist_x, _ = np.histogram(positions_x, bins=8, density=True)
            hist_y, _ = np.histogram(positions_y, bins=8, density=True)

            # Calculate entropy
            entropy_x = entropy(hist_x + 1e-10)  # Add small value to avoid log(0)
            entropy_y = entropy(hist_y + 1e-10)

            return float((entropy_x + entropy_y) / 2)

        except:
            return 0.0

    def _calculate_temporal_entropy(self):
        """Calculate temporal movement entropy"""
        try:
            if len(self.velocity_history) < 8:
                return 0.0

            # Get velocity magnitudes over time
            velocities = []
            for vel_data in self.velocity_history:
                if "left_wrist" in vel_data and "right_wrist" in vel_data:
                    avg_vel = (
                        vel_data["left_wrist"]["magnitude"]
                        + vel_data["right_wrist"]["magnitude"]
                    ) / 2
                    velocities.append(avg_vel)

            if len(velocities) < 8:
                return 0.0

            # Discretize velocities into bins
            hist, _ = np.histogram(velocities, bins=8, density=True)

            # Calculate entropy
            return float(entropy(hist + 1e-10))

        except:
            return 0.0

    def _calculate_arms_shoulders_coordination(self):
        """Calculate coordination between arms and shoulders"""
        try:
            if len(self.arm_positions) < 8 or len(self.shoulder_positions) < 8:
                return 0.0

            # Get arm movement patterns
            arm_movements = []
            shoulder_movements = []

            min_len = min(len(self.arm_positions), len(self.shoulder_positions))

            for i in range(min_len):
                arm_data = self.arm_positions[i]
                shoulder_data = self.shoulder_positions[i]

                # Arm movement magnitude
                arm_center_x = (
                    arm_data["left_wrist"]["x"] + arm_data["right_wrist"]["x"]
                ) / 2
                arm_movements.append(arm_center_x)

                # Shoulder movement magnitude
                shoulder_center_x = (
                    shoulder_data["left"]["x"] + shoulder_data["right"]["x"]
                ) / 2
                shoulder_movements.append(shoulder_center_x)

            if len(arm_movements) > 2 and len(shoulder_movements) > 2:
                correlation = np.corrcoef(arm_movements, shoulder_movements)[0, 1]
                return float(abs(correlation) if not np.isnan(correlation) else 0.0)

            return 0.0

        except:
            return 0.0

    def _calculate_shoulders_hips_coordination(self):
        """Calculate coordination between shoulders and hips"""
        try:
            if len(self.shoulder_positions) < 8 or len(self.hip_positions) < 8:
                return 0.0

            shoulder_movements = []
            hip_movements = []

            min_len = min(len(self.shoulder_positions), len(self.hip_positions))

            for i in range(min_len):
                shoulder_data = self.shoulder_positions[i]
                hip_data = self.hip_positions[i]

                # Shoulder movement
                shoulder_center_x = (
                    shoulder_data["left"]["x"] + shoulder_data["right"]["x"]
                ) / 2
                shoulder_movements.append(shoulder_center_x)

                # Hip movement
                hip_center_x = (hip_data["left"]["x"] + hip_data["right"]["x"]) / 2
                hip_movements.append(hip_center_x)

            if len(shoulder_movements) > 2 and len(hip_movements) > 2:
                correlation = np.corrcoef(shoulder_movements, hip_movements)[0, 1]
                return float(abs(correlation) if not np.isnan(correlation) else 0.0)

            return 0.0

        except:
            return 0.0

    def _calculate_hips_legs_coordination(self):
        """Calculate coordination between hips and legs"""
        try:
            if len(self.hip_positions) < 8 or len(self.knee_positions) < 8:
                return 0.0

            hip_movements = []
            leg_movements = []

            min_len = min(len(self.hip_positions), len(self.knee_positions))

            for i in range(min_len):
                hip_data = self.hip_positions[i]
                knee_data = self.knee_positions[i]

                # Hip movement
                hip_center_x = (hip_data["left"]["x"] + hip_data["right"]["x"]) / 2
                hip_movements.append(hip_center_x)

                # Leg movement (knee)
                knee_center_x = (knee_data["left"]["x"] + knee_data["right"]["x"]) / 2
                leg_movements.append(knee_center_x)

            if len(hip_movements) > 2 and len(leg_movements) > 2:
                correlation = np.corrcoef(hip_movements, leg_movements)[0, 1]
                return float(abs(correlation) if not np.isnan(correlation) else 0.0)

            return 0.0

        except:
            return 0.0

    def _calculate_gait_stability(self):
        """Calculate overall gait stability score"""
        try:
            if len(self.body_center_positions) < 4:
                return 1.0

            positions_x = [pos["pos"]["x"] for pos in self.body_center_positions]
            positions_y = [pos["pos"]["y"] for pos in self.body_center_positions]

            # Calculate stability as inverse of movement variance
            variance_x = np.var(positions_x)
            variance_y = np.var(positions_y)
            total_variance = variance_x + variance_y

            stability = 1.0 / (
                1.0 + total_variance * 100
            )  # Scale factor for sensitivity
            return float(stability)

        except:
            return 1.0

    def _calculate_movement_efficiency(self):
        """Calculate movement efficiency index"""
        try:
            if len(self.velocity_history) < 4:
                return 1.0

            # Get all velocity data
            all_velocities = []
            for vel_data in self.velocity_history:
                for joint_name, vel_info in vel_data.items():
                    if "magnitude" in vel_info:
                        all_velocities.append(vel_info["magnitude"])

            if not all_velocities:
                return 1.0

            # Efficiency as consistency of velocity (inverse of coefficient of variation)
            mean_vel = np.mean(all_velocities)
            std_vel = np.std(all_velocities)

            if mean_vel > 0:
                efficiency = 1.0 / (1.0 + std_vel / mean_vel)
            else:
                efficiency = 1.0

            return float(efficiency)

        except:
            return 1.0

    def _calculate_rhythmic_consistency(self):
        """Calculate rhythmic consistency score"""
        try:
            if len(self.arm_positions) < 12:
                return 1.0

            # Get arm swing pattern
            left_arm_movements = []
            right_arm_movements = []

            for i, arm_data in enumerate(self.arm_positions):
                if i < len(self.shoulder_positions):
                    shoulder_data = self.shoulder_positions[i]

                    left_swing = np.sqrt(
                        (arm_data["left_wrist"]["x"] - shoulder_data["left"]["x"]) ** 2
                        + (arm_data["left_wrist"]["y"] - shoulder_data["left"]["y"])
                        ** 2
                    )
                    right_swing = np.sqrt(
                        (arm_data["right_wrist"]["x"] - shoulder_data["right"]["x"])
                        ** 2
                        + (arm_data["right_wrist"]["y"] - shoulder_data["right"]["y"])
                        ** 2
                    )

                    left_arm_movements.append(left_swing)
                    right_arm_movements.append(right_swing)

            if len(left_arm_movements) < 8 or len(right_arm_movements) < 8:
                return 1.0

            # Calculate rhythmic consistency using autocorrelation
            left_consistency = self._calculate_signal_consistency(left_arm_movements)
            right_consistency = self._calculate_signal_consistency(right_arm_movements)

            return float((left_consistency + right_consistency) / 2)

        except:
            return 1.0

    def _calculate_signal_consistency(self, signal):
        """Calculate signal consistency using autocorrelation"""
        try:
            if len(signal) < 8:
                return 1.0

            signal = np.array(signal)

            # Normalize signal
            signal = (
                (signal - np.mean(signal)) / np.std(signal)
                if np.std(signal) > 0
                else signal
            )

            # Calculate autocorrelation
            autocorr = np.correlate(signal, signal, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Find peak autocorrelation (excluding lag 0)
            if len(autocorr) > 1:
                peak_corr = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 else 0
                return float(max(0, peak_corr))

            return 0.0

        except:
            return 0.0

    def _calculate_signature_features(self, body_size):
        """Calculate unique individual signature features"""
        try:
            features = {}

            # Posture signature (overall body alignment)
            if (
                len(self.shoulder_positions) > 0
                and len(self.hip_positions) > 0
                and len(self.head_positions) > 0
            ):

                posture_scores = []
                min_len = min(
                    len(self.shoulder_positions),
                    len(self.hip_positions),
                    len(self.head_positions),
                )

                for i in range(min_len):
                    shoulder_data = self.shoulder_positions[i]
                    hip_data = self.hip_positions[i]
                    head_data = self.head_positions[i]

                    # Calculate body alignment score
                    shoulder_center_x = (
                        shoulder_data["left"]["x"] + shoulder_data["right"]["x"]
                    ) / 2
                    hip_center_x = (hip_data["left"]["x"] + hip_data["right"]["x"]) / 2
                    head_x = head_data["x"]

                    # Alignment as deviation from vertical
                    alignment_score = (
                        1.0
                        - abs(shoulder_center_x - hip_center_x)
                        - abs(head_x - shoulder_center_x)
                    )
                    posture_scores.append(max(0, alignment_score))

                features["posture_signature"] = (
                    float(np.mean(posture_scores)) if posture_scores else 0.5
                )
            else:
                features["posture_signature"] = 0.5

            # Movement style index (combination of smoothness and amplitude)
            if len(self.arm_positions) > 4:
                arm_amplitudes = []

                for i, arm_data in enumerate(self.arm_positions):
                    if i < len(self.shoulder_positions):
                        shoulder_data = self.shoulder_positions[i]

                        # Calculate total arm movement amplitude
                        left_amplitude = np.sqrt(
                            (arm_data["left_wrist"]["x"] - shoulder_data["left"]["x"])
                            ** 2
                            + (arm_data["left_wrist"]["y"] - shoulder_data["left"]["y"])
                            ** 2
                        )
                        right_amplitude = np.sqrt(
                            (arm_data["right_wrist"]["x"] - shoulder_data["right"]["x"])
                            ** 2
                            + (
                                arm_data["right_wrist"]["y"]
                                - shoulder_data["right"]["y"]
                            )
                            ** 2
                        )

                        arm_amplitudes.append((left_amplitude + right_amplitude) / 2)

                if arm_amplitudes:
                    amplitude_consistency = (
                        1.0 / (1.0 + np.std(arm_amplitudes) / np.mean(arm_amplitudes))
                        if np.mean(arm_amplitudes) > 0
                        else 0
                    )
                    features["movement_style_index"] = float(amplitude_consistency)
                else:
                    features["movement_style_index"] = 0.5
            else:
                features["movement_style_index"] = 0.5

            # Energy distribution pattern
            features["energy_distribution_pattern"] = (
                self._calculate_energy_distribution()
            )

            # Bilateral coordination quality
            features["bilateral_coordination_quality"] = (
                self._calculate_bilateral_coordination()
            )

            # Movement complexity index
            features["movement_complexity_index"] = (
                self._calculate_movement_complexity()
            )

            # Personal rhythm signature
            features["personal_rhythm_signature"] = self._calculate_personal_rhythm()

            # Biomechanical efficiency
            features["biomechanical_efficiency"] = (
                self._calculate_biomechanical_efficiency()
            )

            # Postural sway signature
            features["postural_sway_signature"] = (
                self._calculate_postural_sway_signature(body_size)
            )

            return features

        except Exception as e:
            print(f"[GaitProcessor] Error calculating signature features: {str(e)}")
            return {}

    def _calculate_energy_distribution(self):
        """ energy distribution across body segments"""
        try:
            if not self.velocity_history:
                return 0.5

            #  energy for different body segments
            arm_energy = []
            shoulder_energy = []
            hip_energy = []

            for vel_data in self.velocity_history:
                # Arm energy
                arm_vel = 0
                if "left_wrist" in vel_data and "right_wrist" in vel_data:
                    arm_vel = (
                        vel_data["left_wrist"]["magnitude"]
                        + vel_data["right_wrist"]["magnitude"]
                    ) / 2
                arm_energy.append(arm_vel**2)

                # Shoulder energy
                shoulder_vel = 0
                if "left_shoulder" in vel_data and "right_shoulder" in vel_data:
                    shoulder_vel = (
                        vel_data["left_shoulder"]["magnitude"]
                        + vel_data["right_shoulder"]["magnitude"]
                    ) / 2
                shoulder_energy.append(shoulder_vel**2)

                # Hip energy
                hip_vel = 0
                if "left_hip" in vel_data and "right_hip" in vel_data:
                    hip_vel = (
                        vel_data["left_hip"]["magnitude"]
                        + vel_data["right_hip"]["magnitude"]
                    ) / 2
                hip_energy.append(hip_vel**2)

            # Calculate energy distribution
            total_arm_energy = np.sum(arm_energy) if arm_energy else 0
            total_shoulder_energy = np.sum(shoulder_energy) if shoulder_energy else 0
            total_hip_energy = np.sum(hip_energy) if hip_energy else 0

            total_energy = total_arm_energy + total_shoulder_energy + total_hip_energy

            if total_energy > 0:
                # Energy distribution as entropy
                energies = (
                    np.array(
                        [total_arm_energy, total_shoulder_energy, total_hip_energy]
                    )
                    / total_energy
                )
                distribution = entropy(energies + 1e-10)
                return float(distribution)

            return 0.5

        except:
            return 0.5

    def _calculate_bilateral_coordination(self):
        try:
            if len(self.arm_positions) < 6:
                return 0.5

            coordination_scores = []

            # Compare left and right movements
            for i, arm_data in enumerate(self.arm_positions):
                if i < len(self.shoulder_positions):
                    shoulder_data = self.shoulder_positions[i]

                    # Left side movement
                    left_movement = np.sqrt(
                        (arm_data["left_wrist"]["x"] - shoulder_data["left"]["x"]) ** 2
                        + (arm_data["left_wrist"]["y"] - shoulder_data["left"]["y"])
                        ** 2
                    )

                    # Right side movement
                    right_movement = np.sqrt(
                        (arm_data["right_wrist"]["x"] - shoulder_data["right"]["x"])
                        ** 2
                        + (arm_data["right_wrist"]["y"] - shoulder_data["right"]["y"])
                        ** 2
                    )

                    # Coordination as similarity between sides
                    if left_movement + right_movement > 0:
                        coordination = 1.0 - abs(left_movement - right_movement) / (
                            left_movement + right_movement
                        )
                        coordination_scores.append(max(0, coordination))

            return float(np.mean(coordination_scores)) if coordination_scores else 0.5

        except:
            return 0.5

    def _calculate_movement_complexity(self):
        try:
            if len(self.body_center_positions) < 8:
                return 0.5

            positions_x = [pos["pos"]["x"] for pos in self.body_center_positions]
            positions_y = [pos["pos"]["y"] for pos in self.body_center_positions]

            # Calculate complexity using fractal dimension approximation
            complexity_x = self._approximate_fractal_dimension(positions_x)
            complexity_y = self._approximate_fractal_dimension(positions_y)

            return float((complexity_x + complexity_y) / 2)

        except:
            return 0.5

    def _approximate_fractal_dimension(self, trajectory):
        """Approximate fractal dimension of trajectory"""
        try:
            if len(trajectory) < 4:
                return 1.0

            trajectory = np.array(trajectory)

            # Calculate path length at different scales
            scales = [1, 2, 4]
            lengths = []

            for scale in scales:
                if len(trajectory) > scale:
                    downsampled = trajectory[::scale]
                    if len(downsampled) > 1:
                        path_length = np.sum(np.abs(np.diff(downsampled)))
                        lengths.append(path_length)

            if len(lengths) >= 2:
                # Estimate fractal dimension
                log_scales = np.log(scales[: len(lengths)])
                log_lengths = np.log(np.array(lengths) + 1e-10)

                # Linear regression
                slope = np.polyfit(log_scales, log_lengths, 1)[0]
                fractal_dim = 1 - slope

                return float(max(1.0, min(2.0, fractal_dim)))

            return 1.0

        except:
            return 1.0

    def _calculate_personal_rhythm(self):
        try:
            if len(self.arm_positions) < 12:
                return 0.5

            # rhythmic pattern from arm swings
            left_arm_pattern = []
            right_arm_pattern = []

            for i, arm_data in enumerate(self.arm_positions):
                if i < len(self.shoulder_positions):
                    shoulder_data = self.shoulder_positions[i]

                    left_swing = (
                        arm_data["left_wrist"]["x"] - shoulder_data["left"]["x"]
                    )
                    right_swing = (
                        arm_data["right_wrist"]["x"] - shoulder_data["right"]["x"]
                    )

                    left_arm_pattern.append(left_swing)
                    right_arm_pattern.append(right_swing)

            if len(left_arm_pattern) > 8:
                # rhythm strength FFT
                fft_left = np.fft.fft(left_arm_pattern)
                fft_right = np.fft.fft(right_arm_pattern)

                power_left = np.abs(fft_left) ** 2
                power_right = np.abs(fft_right) ** 2

                if len(power_left) > 2 and len(power_right) > 2:
                    rhythm_strength_left = (
                        np.max(power_left[1:]) / np.sum(power_left[1:])
                        if np.sum(power_left[1:]) > 0
                        else 0
                    )
                    rhythm_strength_right = (
                        np.max(power_right[1:]) / np.sum(power_right[1:])
                        if np.sum(power_right[1:]) > 0
                        else 0
                    )

                    return float((rhythm_strength_left + rhythm_strength_right) / 2)

            return 0.5

        except:
            return 0.5

    def _calculate_biomechanical_efficiency(self):
        try:
            if not self.velocity_history or not self.acceleration_history:
                return 0.5

            smooth_motion = 0
            total_motion = 0

            for vel_data in self.velocity_history:
                for joint_name, vel_info in vel_data.items():
                    if "magnitude" in vel_info:
                        total_motion += vel_info["magnitude"]

            for acc_data in self.acceleration_history:
                for joint_name, acc_info in acc_data.items():
                    if "magnitude" in acc_info:
                        if acc_info["magnitude"] < np.mean(
                            [
                                a.get(joint_name, {}).get("magnitude", 0)
                                for a in self.acceleration_history
                            ]
                        ):
                            smooth_motion += 1

            if total_motion > 0:
                efficiency = (
                    smooth_motion
                    / len(self.acceleration_history)
                    / len(self.velocity_history[0])
                    if self.velocity_history
                    else 0
                )
                return float(min(1.0, efficiency))

            return 0.5

        except:
            return 0.5

    def _calculate_postural_sway_signature(self, body_size):
        try:
            if len(self.body_center_positions) < 8:
                return 0.5

            positions_x = [pos["pos"]["x"] for pos in self.body_center_positions]
            positions_y = [pos["pos"]["y"] for pos in self.body_center_positions]

            # Normalize by body size
            positions_x = (
                np.array(positions_x) / body_size
                if body_size > 0
                else np.array(positions_x)
            )
            positions_y = (
                np.array(positions_y) / body_size
                if body_size > 0
                else np.array(positions_y)
            )

            # Calculate sway characteristics
            sway_area = np.std(positions_x) * np.std(positions_y)
            sway_velocity = np.mean(
                np.sqrt(np.diff(positions_x) ** 2 + np.diff(positions_y) ** 2)
            )

            signature = sway_area * (1 + sway_velocity)
            return float(min(1.0, signature))

        except:
            return 0.5

    def _validate_enhanced_features(self, features):
        try:
            if not features:
                return False

            if len(features) < len(self.feature_names) * 0.8:  # Allow 80% of features
                return False

            invalid_count = 0
            for feature_name, value in features.items():
                if (
                    not isinstance(value, (int, float))
                    or np.isnan(value)
                    or np.isinf(value)
                ):
                    invalid_count += 1
                    if (
                        invalid_count > len(features) * 0.2
                    ):  # Allow up to 20% invalid values
                        return False

            stability_score = self._calculate_feature_stability(features)
            return (
                stability_score >= self.feature_stability_threshold * 0.6
            )  

        except Exception as e:
            print(f"[GaitProcessor] Error validating features: {str(e)}")
            return True

    def _calculate_feature_stability(self, features):
        try:
            if len(self.feature_history) < 2:
                self.feature_history.append(features)
                return 1.0

            prev_features = self.feature_history[-1]
            stability_scores = []

            for feature_name in features:
                if feature_name in prev_features:
                    current_val = features[feature_name]
                    prev_val = prev_features[feature_name]

                    if prev_val != 0:
                        stability = 1.0 - abs(current_val - prev_val) / (
                            abs(prev_val) + 1e-6
                        )
                        stability_scores.append(max(0, stability))
                    else:
                        stability_scores.append(1.0 if current_val == 0 else 0.7)

            avg_stability = np.mean(stability_scores) if stability_scores else 0.7

            self.feature_history.append(features)

            return float(avg_stability)

        except:
            return 0.7

    def _update_signature_tracking(self, features):
        try:
            for feature_name, value in features.items():
                if feature_name not in self.signature_features:
                    self.signature_features[feature_name] = []

                self.signature_features[feature_name].append(value)

                if (
                    len(self.signature_features[feature_name])
                    > self.max_signature_history
                ):
                    self.signature_features[feature_name] = self.signature_features[
                        feature_name
                    ][-self.max_signature_history :]

                if len(self.signature_features[feature_name]) > 3:
                    recent_values = self.signature_features[feature_name][-5:]
                    stability = 1.0 / (
                        1.0 + np.std(recent_values) / (np.mean(recent_values) + 1e-6)
                    )
                    self.feature_stability_scores[feature_name] = stability

            self.frame_counter += 1

            if self.frame_counter >= self.memory_cleanup_threshold:
                self._perform_memory_cleanup()
                self.frame_counter = 0

        except Exception as e:
            print(f"[GaitProcessor] Error updating signature tracking: {str(e)}")

    def _perform_memory_cleanup(self):
        try:
            if len(self.feature_history) > self.max_feature_history:
                self.feature_history = deque(
                    list(self.feature_history)[-self.max_feature_history :],
                    maxlen=self.max_feature_history,
                )

            for feature_name in list(self.signature_features.keys()):
                if (
                    len(self.signature_features[feature_name])
                    > self.max_signature_history
                ):
                    self.signature_features[feature_name] = self.signature_features[
                        feature_name
                    ][-self.max_signature_history :]

            old_features = set(self.feature_stability_scores.keys()) - set(
                self.signature_features.keys()
            )
            for feature in old_features:
                del self.feature_stability_scores[feature]

            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GaitProcessor] Error during memory cleanup: {str(e)}")

    def get_feature_vector(self, landmarks):
        features = self.extract_gait_features(landmarks)
        if features:
            return np.array([features.get(name, 0.0) for name in self.feature_names])
        return None

    def get_signature_summary(self):
        try:
            summary = {}

            for feature_name in self.feature_names:
                if (
                    feature_name in self.signature_features
                    and len(self.signature_features[feature_name]) > 0
                ):
                    values = self.signature_features[feature_name]
                    summary[feature_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "stability": self.feature_stability_scores.get(
                            feature_name, 0.0
                        ),
                        "sample_count": len(values),
                    }

            return summary

        except Exception as e:
            print(f"[GaitProcessor] Error getting signature summary: {str(e)}")
            return {}

    def reset_buffers(self):
        try:
            self._initialize_buffers()

            self.signature_features.clear()
            self.feature_stability_scores.clear()
            self.feature_history.clear()

            self.frame_counter = 0
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[GaitProcessor] Buffers reset and memory cleaned")

        except Exception as e:
            print(f"[GaitProcessor] Error resetting buffers: {str(e)}")

    def set_quality_level(self, quality_level):
        if quality_level in self.quality_levels:
            self.current_quality = quality_level
            self.update_quality_settings()
            print(f"[GaitProcessor] Quality level set to: {quality_level}")
        else:
            print(f"[GaitProcessor] Invalid quality level: {quality_level}")

    def get_processing_stats(self):
        return {
            "total_features": len(self.feature_names),
            "buffer_sizes": {
                "arm_positions": len(self.arm_positions),
                "shoulder_positions": len(self.shoulder_positions),
                "head_positions": len(self.head_positions),
                "hip_positions": len(self.hip_positions),
                "knee_positions": len(self.knee_positions),
                "ankle_positions": len(self.ankle_positions),
                "velocity_history": len(self.velocity_history),
                "acceleration_history": len(self.acceleration_history),
            },
            "current_quality": self.current_quality,
            "signature_features_tracked": len(self.signature_features),
            "feature_stability_avg": (
                np.mean(list(self.feature_stability_scores.values()))
                if self.feature_stability_scores
                else 0.0
            ),
        }

    def _get_feature_weights(self):
        weights = {
            # Upper Body
            "arm_swing_asymmetry_mean": 1.0,
            "arm_swing_asymmetry_std": 0.9,
            "arm_swing_amplitude_left": 1.0,
            "arm_swing_amplitude_right": 1.0,
            "arm_swing_velocity_ratio": 0.9,
            "arm_swing_frequency_left": 0.8,
            "arm_swing_frequency_right": 0.8,
            "arm_swing_phase_lag": 0.9,
            "arm_swing_regularity_index": 0.9,
            "elbow_bend_asymmetry": 0.8,
            "wrist_trajectory_complexity": 0.7,
            "upper_body_lean_angle": 0.9,
            "shoulder_elbow_coordination": 0.8,
            "upper_body_stability_index": 0.9,
            "arm_swing_symmetry_index": 0.8,
            "shoulder_height_variation": 0.7,
            "upper_body_rhythm_consistency": 0.9,
            # Shoulder Dynamics
            "shoulder_roll_mean": 0.8,
            "shoulder_roll_std": 0.7,
            "shoulder_roll_asymmetry": 0.8,
            "shoulder_width_variation_coeff": 0.7,
            "shoulder_bounce_amplitude": 0.8,
            "shoulder_bounce_frequency": 0.7,
            "shoulder_sway_lateral_mean": 0.8,
            "shoulder_sway_lateral_std": 0.7,
            "shoulder_velocity_profile": 0.8,
            "shoulder_acceleration_pattern": 0.8,
            # Head Movement
            "head_bob_amplitude": 0.6,
            "head_bob_frequency": 0.5,
            "head_sway_lateral_mean": 0.6,
            "head_sway_pattern_regularity": 0.6,
            "head_stability_index": 0.7,
            "head_trajectory_smoothness": 0.6,
            "head_shoulder_coordination": 0.7,
            "neck_angle_variation": 0.6,
            # Lower Body Features
            "knee_separation_mean": 0.9,
            "knee_separation_std": 0.8,
            "knee_separation_rhythm": 0.9,
            "stance_width_variation_coeff": 0.8,
            # Coordination Features
            "movement_smoothness_global": 0.9,
            "movement_smoothness_upper": 0.9,
            "vertical_bounce_amplitude": 0.8,
            "vertical_bounce_frequency": 0.7,
            "movement_entropy_spatial": 0.8,
            "movement_entropy_temporal": 0.8,
            "coordination_index_arms_shoulders": 0.9,
            "gait_stability_score": 0.9,
            "movement_efficiency_index": 0.8,
            "rhythmic_consistency_score": 0.9,
            # Signature Features
            "posture_signature": 0.9,
            "movement_style_index": 0.9,
            "energy_distribution_pattern": 0.8,
            "bilateral_coordination_quality": 0.9,
            "movement_complexity_index": 0.8,
            "personal_rhythm_signature": 0.9,
            "biomechanical_efficiency": 0.8,
            "postural_sway_signature": 0.8,
        }

        # Normalize weights
        max_weight = max(weights.values())
        return {k: v / max_weight for k, v in weights.items()}

    def __del__(self):
        try:
            self.reset_buffers()
        except:
            pass
