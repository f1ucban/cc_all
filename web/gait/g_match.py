import torch
import numpy as np
from web.database import db

from web.gait.g_proc import GaitProcessor


class GaitMatcher:
    def __init__(self, pose_processor, face_matcher):
        self.pose_processor = pose_processor
        self.face_matcher = face_matcher
        self.gait_processor = GaitProcessor()

        self.feature_names = self.gait_processor.feature_names
        self.feature_weights = self.gait_processor._get_feature_weights()

        self.quality_levels = self.gait_processor.quality_levels
        self.current_quality = "high"
        self.min_frames = self.quality_levels[self.current_quality][
            "min_frames_for_analysis"
        ]

        self.gait_threshold = 0.3
        self.fusion_weight = 0.5
        self.min_confidence = 0.45

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_history = []
        self.last_fused_features = None

        self.feature_groups = {
            "Upper Body Features": slice(0, 17),
            "Shoulder Dynamics": slice(17, 27),
            "Head Movement Patterns": slice(27, 35),
            "Lower Body Features": slice(35, 39),
            "Coordination Features": slice(39, 49),
            "Individual Signature Features": slice(49, 57),
        }

        print(
            f"[GaitMatcher] Initialized with {len(self.feature_names)} discriminative features"
        )
        print(
            f"[GaitMatcher] Using GaitProcessor's feature weights and quality settings"
        )

    def match_gait(self, features, user_idx=None):
        try:
            if not features:
                print("[GaitMatcher]", "No features provided for matching")
                return None

            print("[GaitMatcher]", "Starting gait matching process...")
            print("[GaitMatcher]", f"Input features: {features}")

            if isinstance(features, list):
                feature_arrays = []
                for feat_dict in features:
                    if isinstance(feat_dict, dict):
                        feature_array = np.array(
                            [feat_dict[name] for name in self.feature_names],
                            dtype=np.float32,
                        )
                        feature_arrays.append(feature_array)

                if feature_arrays:
                    feature_array = np.mean(feature_arrays, axis=0)
                    print("[GaitMatcher]", f"Averaged features: {feature_array}")
                else:
                    print("[GaitMatcher]", "No valid features found in input list")
                    return None
            elif isinstance(features, dict):
                feature_array = np.array(
                    [features[name] for name in self.feature_names], dtype=np.float32
                )
            else:
                feature_array = features

            if user_idx is None:
                users = db.get_all_users()
                print("[GaitMatcher]", f"Found {len(users)} users in database")
            else:
                users = [db.get_user_by_idx(user_idx)]
                if not users[0]:
                    print("[GaitMatcher]", f"No user found with idx {user_idx}")
                    return None

            best_match = None
            best_score = -1
            all_scores = []

            for user in users:
                if not user:
                    continue

                print(
                    "[GaitMatcher]",
                    f"\nChecking user: {user['firstname']} {user['lastname']} (ID: {user['user_idx']})",
                )

                enrollments = db.get_user_gait_enrollments(user["user_idx"])
                print(
                    "[GaitMatcher]",
                    f"Found {len(enrollments)} enrollments for this user",
                )

                if not enrollments:
                    print(
                        "[GaitMatcher]",
                        "No gait enrollments found for this user - skipping",
                    )
                    continue

                enrollment_scores = []
                for i, enrollment in enumerate(enrollments):
                    if not enrollment["gait_features"]:
                        continue

                    stored_features = np.frombuffer(
                        enrollment["gait_features"], dtype=np.float32
                    )
                    print("[GaitMatcher]", f"\nEnrollment {i+1}:")
                    print("[GaitMatcher]", f"Stored features: {stored_features}")

                    similarity = self._calculate_weighted_similarity(
                        feature_array, stored_features
                    )
                    print(
                        "[GaitMatcher]",
                        f"Similarity scores (base, sigmoid): {similarity}",
                    )
                    enrollment_scores.append(similarity[1])

                if enrollment_scores:
                    avg_score = np.mean(enrollment_scores)
                    print("[GaitMatcher]", f"Average score for user: {avg_score:.3f}")
                    all_scores.append((user, avg_score))

                    if avg_score > best_score:
                        best_score = avg_score
                        best_match = {
                            "user_idx": user["user_idx"],
                            "firstname": user["firstname"],
                            "lastname": user["lastname"],
                            "role": user["role"],
                            "confidence": float(avg_score),
                        }
                        print("[GaitMatcher]", f"New best match found: {best_match}")

            if best_match and best_match["confidence"] >= self.gait_threshold:
                print(
                    "[GaitMatcher]",
                    f"\nFinal match found: {best_match['firstname']} {best_match['lastname']} with score {best_match['confidence']}",
                )
                return best_match
            else:
                print("[GaitMatcher]", "\nNo match found above threshold")
                if best_match:
                    print(
                        "[GaitMatcher]",
                        f"Best score was {best_match['confidence']} (threshold: {self.gait_threshold})",
                    )
            return None

        except Exception as e:
            print("[GaitMatcher]", f"Error in match_gait: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def _calculate_weighted_similarity(self, features1, features2):
        try:
            features1 = np.array(features1, dtype=np.float32)
            features2 = np.array(features2, dtype=np.float32)

            print("\n[GaitMatcher] Detailed Feature Analysis:")
            print("----------------------------------------")
            print(f"Input Features Shape: {features1.shape}")
            print(f"Feature Range Check:")
            print(
                f"Features1 - Min: {np.min(features1):.4f}, Max: {np.max(features1):.4f}, Mean: {np.mean(features1):.4f}"
            )
            print(
                f"Features2 - Min: {np.min(features2):.4f}, Max: {np.max(features2):.4f}, Mean: {np.mean(features2):.4f}"
            )

            feature_groups = self.feature_groups
            group_scores = {}
            diffs = []

            for group_name, group_slice in feature_groups.items():
                print(f"\n[GaitMatcher] {group_name}:")
                print("----------------------------------------")
                group_features1 = features1[group_slice]
                group_features2 = features2[group_slice]

                print(f"Group Statistics:")
                print(
                    f"Features1 - Min: {np.min(group_features1):.4f}, Max: {np.max(group_features1):.4f}, Mean: {np.mean(group_features1):.4f}"
                )
                print(
                    f"Features2 - Min: {np.min(group_features2):.4f}, Max: {np.max(group_features2):.4f}, Mean: {np.mean(group_features2):.4f}"
                )

                group_diffs = []
                for i, (name, weight) in enumerate(
                    list(self.feature_weights.items())[group_slice]
                ):
                    val1 = group_features1[i]
                    val2 = group_features2[i]

                    if abs(val1) < 1e-6 and abs(val2) < 1e-6:
                        diff = 0.0
                        print(
                            f"{name}: Zero values detected (val1: {val1:.4f}, val2: {val2:.4f})"
                        )
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            # Combine relative and absolute differences
                            rel_diff = abs(val1 - val2) / max_val
                            abs_diff = abs(val1 - val2)

                            if "asymmetry" in name or "variation" in name:
                                # Tolerance for asymmetry measures
                                diff = 0.8 * rel_diff + 0.2 * min(1.0, abs_diff)
                                print(
                                    f"{name}: Asymmetry feature - Rel: {rel_diff:.4f}, Abs: {abs_diff:.4f}"
                                )
                            elif "frequency" in name or "rhythm" in name:
                                # Frequency features
                                diff = 0.9 * rel_diff + 0.1 * min(1.0, abs_diff)
                                print(
                                    f"{name}: Frequency feature - Rel: {rel_diff:.4f}, Abs: {abs_diff:.4f}"
                                )
                            elif "stability" in name or "smoothness" in name:
                                # Strict stability measures
                                diff = 0.7 * rel_diff + 0.3 * min(1.0, abs_diff)
                                print(
                                    f"{name}: Stability feature - Rel: {rel_diff:.4f}, Abs: {abs_diff:.4f}"
                                )
                            elif "signature" in name or "pattern" in name:
                                # Individual signature features
                                diff = 0.85 * rel_diff + 0.15 * min(1.0, abs_diff)
                                print(
                                    f"{name}: Signature feature - Rel: {rel_diff:.4f}, Abs: {abs_diff:.4f}"
                                )
                            else:
                                # Default weighting
                                diff = 0.85 * rel_diff + 0.15 * min(1.0, abs_diff)
                                print(
                                    f"{name}: Standard feature - Rel: {rel_diff:.4f}, Abs: {abs_diff:.4f}"
                                )
                        else:
                            diff = 0.0
                            print(f"{name}: Zero max value detected")

                    weighted_diff = diff * weight
                    group_diffs.append(weighted_diff)
                    diffs.append(weighted_diff)
                    print(
                        f"Final {name}: {diff:.4f} (weight: {weight:.2f}, weighted: {weighted_diff:.4f})"
                    )

                group_scores[group_name] = 1.0 - sum(group_diffs) / sum(
                    list(self.feature_weights.values())[group_slice]
                )
                print(f"\n{group_name} Score: {group_scores[group_name]:.4f}")

            total_diff = sum(diffs) / sum(self.feature_weights.values())
            base_score = 1.0 - total_diff

            # sigmoid transformation
            k = 6.0  # steepness
            x0 = 0.6  # midpoint
            sigmoid_score = 1 / (1 + np.exp(-k * (base_score - x0)))

            # scaling
            sigmoid_score = (
                0.7 + (sigmoid_score - 0.7) * 0.3
            )  # Scale range to [0.7, 1.0]

            print("\n[GaitMatcher] Final Scores:")
            print("----------------------------------------")
            print(f"Group Scores: {group_scores}")
            print(f"Base Score: {base_score:.4f}")
            print(f"Final Sigmoid Score: {sigmoid_score:.4f}")

            return base_score, sigmoid_score

        except Exception as e:
            print("[GaitMatcher]", f"Error in similarity calculation: {str(e)}")
            return 0.0, 0.0

    def fuse_recognition(self, face_results, gait_result):
        if not face_results and not gait_result:
            print("[GaitMatcher]", "No recognition results available for fusion")
            return None

        if not face_results:
            print("[GaitMatcher]", "Using gait-only recognition result")
            return gait_result

        if not gait_result:
            print("[GaitMatcher]", "Using face-only recognition result")
            return face_results[0]

        print("[GaitMatcher]", "Fusing face and gait recognition results")

        matching_face = next(
            (
                face
                for face in face_results
                if face.get("user_idx") == gait_result.get("user_idx")
            ),
            None,
        )

        if matching_face:
            face_embed = matching_face.get("embedding")
            gait_features = gait_result.get("features")

            if face_embed is not None and gait_features is not None:
                face_embed_norm = face_embed / np.linalg.norm(face_embed)

                gait_features_norm = np.array(
                    [gait_features[f] for f in self.feature_names]
                )
                gait_features_norm = gait_features_norm / np.linalg.norm(
                    gait_features_norm
                )

                current_fused_features = np.concatenate(
                    [face_embed_norm, gait_features_norm]
                )

                user_idx = matching_face.get("user_idx")
                stored_fused_features = None

                try:
                    fused_data = db.get_fused_features()
                    user_fused = next(
                        (f for f in fused_data if f["user_idx"] == user_idx), None
                    )
                    if user_fused and user_fused["fused_features"]:
                        stored_fused_features = np.frombuffer(
                            user_fused["fused_features"], dtype=np.float32
                        )
                except Exception as e:
                    print("[GaitMatcher]", f"Error retrieving fused features: {str(e)}")

                if stored_fused_features is not None:
                    stored_fused_features = stored_fused_features / np.linalg.norm(
                        stored_fused_features
                    )
                    similarity = np.dot(current_fused_features, stored_fused_features)
                else:
                    similarity = (
                        1
                        - np.linalg.norm(
                            current_fused_features - self.last_fused_features
                        )
                        if hasattr(self, "last_fused_features")
                        else 0.5
                    )

                self.last_fused_features = current_fused_features

                if similarity > 0.9:
                    try:
                        features_bytes = current_fused_features.astype(
                            np.float32
                        ).tobytes()
                        db.update_fused_features(user_idx, features_bytes)
                        print("[GaitMatcher]", "Updated stored fused features")
                    except Exception as e:
                        print(
                            "[GaitMatcher]", f"Error updating fused features: {str(e)}"
                        )

                face_confidence = matching_face.get("confidence", 0.0)
                gait_confidence = gait_result.get("confidence", 0.0)
                fused_confidence = (
                    self.fusion_weight * face_confidence
                    + (1 - self.fusion_weight) * gait_confidence
                )

                return {
                    "user_idx": user_idx,
                    "firstname": matching_face.get("firstname"),
                    "lastname": matching_face.get("lastname"),
                    "role": matching_face.get("role"),
                    "confidence": float(fused_confidence),
                    "face_confidence": float(face_confidence),
                    "gait_confidence": float(gait_confidence),
                    "type": "fusion",
                    "features": current_fused_features.tolist(),
                }

        print(
            "[GaitMatcher]",
            "Face match not found for gait-matched user. Returning gait result.",
        )
        return gait_result

    def _process_input_features(self, gait_features):
        try:
            if not self.gait_processor._validate_enhanced_features(gait_features):
                print("[GaitMatcher] Invalid feature format")
                return None

            return np.array(
                [gait_features[name] for name in self.feature_names],
                dtype=np.float32,
            )

        except Exception as e:
            print(f"[GaitMatcher] Error processing input features: {str(e)}")
            return None

    def enroll_gait(self, user_idx, gait_features):
        """Enroll new gait features for a user"""
        if not gait_features:
            print("[GaitMatcher]", "No gait features provided")
            return False

        try:
            feature_array = self._process_input_features(gait_features)
            if feature_array is None:
                return False

            if len(feature_array) != len(self.feature_names):
                print(
                    "[GaitMatcher]",
                    f"Expected {len(self.feature_names)} features, got {len(feature_array)}",
                )
                return False

            print("[GaitMatcher]", "Feature verification during enrollment:")
            for name, value in zip(self.feature_names, feature_array):
                print(f"{name}: {value:.4f}")

            # Convert to bytes for storage
            features_bytes = feature_array.astype(np.float32).tobytes()

            # Update database
            success = db.update_gait_features(user_idx, features_bytes)
            if not success:
                print("[GaitMatcher]", "Failed to update gait features in database")
                return False

            print(
                "[GaitMatcher]",
                f"Successfully enrolled gait features for user {user_idx}",
            )
            return True

        except Exception as e:
            print("[GaitMatcher]", f"Error enrolling gait features: {str(e)}")
            return False

    def update_fusion_weight(self, new_weight):
        if 0.0 <= new_weight <= 1.0:
            self.fusion_weight = new_weight
            print("[GaitMatcher]", f"Updated fusion weight to {new_weight}")
            return True
        else:
            print("[GaitMatcher]", "Fusion weight must be between 0 and 1")
            return False

    def update_thresholds(self, gait_threshold=None, min_confidence=None):
        """Update recognition thresholds"""
        if gait_threshold is not None:
            if 0.0 <= gait_threshold <= 1.0:
                self.gait_threshold = gait_threshold
                print("[GaitMatcher]", f"Updated gait threshold to {gait_threshold}")
            else:
                print("[GaitMatcher]", "Gait threshold must be between 0 and 1")

        if min_confidence is not None:
            if 0.0 <= min_confidence <= 1.0:
                self.min_confidence = min_confidence
                print(
                    "[GaitMatcher]", f"Updated minimum confidence to {min_confidence}"
                )
            else:
                print("[GaitMatcher]", "Minimum confidence must be between 0 and 1")

    def reset(self):
        """Reset the matcher state"""
        if hasattr(self, "last_fused_features"):
            del self.last_fused_features
        print("[GaitMatcher]", "State reset")

    def verify_match_quality(self, test_features, enrolled_features):
        try:
            print("\n[GaitMatcher] Quality Verification:")
            print("----------------------------------------")

            if isinstance(test_features, dict):
                test_tensor = torch.tensor(
                    [test_features[name] for name in self.feature_names],
                    device=self.device,
                )
                print("Test features converted from dictionary")
            else:
                test_tensor = (
                    test_features.to(self.device)
                    if not test_features.is_cuda
                    else test_features
                )
                print("Test features already in tensor format")

            if isinstance(enrolled_features, dict):
                enrolled_tensor = torch.tensor(
                    [enrolled_features[name] for name in self.feature_names],
                    device=self.device,
                )
                print("Enrolled features converted from dictionary")
            else:
                enrolled_tensor = (
                    enrolled_features.to(self.device)
                    if not enrolled_features.is_cuda
                    else enrolled_features
                )
                print("Enrolled features already in tensor format")

            print("\nFeature:")
            print("----------------------------------------")
            print(
                f"Test Features - Min: {torch.min(test_tensor):.4f}, Max: {torch.max(test_tensor):.4f}, Mean: {torch.mean(test_tensor):.4f}"
            )
            print(
                f"Enrolled Features - Min: {torch.min(enrolled_tensor):.4f}, Max: {torch.max(enrolled_tensor):.4f}, Mean: {torch.mean(enrolled_tensor):.4f}"
            )

            differences = torch.abs(test_tensor - enrolled_tensor)
            print(
                f"\nDifference - Min: {torch.min(differences):.4f}, Max: {torch.max(differences):.4f}, Mean: {torch.mean(differences):.4f}"
            )

            stability_score = self.gait_processor._calculate_feature_stability(
                {name: float(val) for name, val in zip(self.feature_names, test_tensor)}
            )

            return (
                stability_score >= self.gait_processor.feature_stability_threshold * 0.6
            )

        except Exception as e:
            print(f"[GaitMatcher] Error verifying match quality: {str(e)}")
            return False

    def match_features(self, test_features, enrolled_features):
        try:
            if isinstance(test_features, dict):
                test_tensor = torch.tensor(
                    [test_features[name] for name in self.feature_names],
                    device=self.device,
                )
            else:
                test_tensor = (
                    test_features.to(self.device)
                    if not test_features.is_cuda
                    else test_features
                )

            if isinstance(enrolled_features, dict):
                enrolled_tensor = torch.tensor(
                    [enrolled_features[name] for name in self.feature_names],
                    device=self.device,
                )
            else:
                enrolled_tensor = (
                    enrolled_features.to(self.device)
                    if not enrolled_features.is_cuda
                    else enrolled_features
                )

            similarity = self.calculate_weighted_similarity(
                test_tensor, enrolled_tensor
            )

            differences = torch.abs(test_tensor - enrolled_tensor)
            feature_diffs = {
                name: float(diff) for name, diff in zip(self.feature_names, differences)
            }

            print("\n[GaitMatcher] Feature-wise Differences:")
            for name, diff in feature_diffs.items():
                print(f"{name}: {diff:.4f}")

            print(f"\n[GaitMatcher] Overall Similarity: {similarity:.4f}")

            is_quality_match = self.verify_match_quality(
                test_features, enrolled_features
            )

            return similarity, is_quality_match

        except Exception as e:
            print(f"[GaitMatcher] Error in feature matching: {str(e)}")
            return 0.0, False

    def set_quality_level(self, quality_level):
        if quality_level in self.quality_levels:
            self.current_quality = quality_level
            self.min_frames = self.quality_levels[quality_level][
                "min_frames_for_analysis"
            ]
            print(f"[GaitMatcher] Quality level set to: {quality_level}")
            print(f"[GaitMatcher] Min frames updated to: {self.min_frames}")
        else:
            print(f"[GaitMatcher] Invalid quality level: {quality_level}")
