import cv2 as cv
import numpy as np
from datetime import datetime
from scipy.optimize import linear_sum_assignment


from web.database import db
from web.utils.config import known, unknown


class FaceMatcher:
    def __init__(self, proc):
        self.proc = proc
        self.last_log = {}
        self.rec_hist = {}
        self.hist_win = 10
        self.pos_hist = {}
        self.pos_thresh = 30
        self.last_ids = {}
        self.id_conf = {}
        self.id_stab = {}
        self.stab_thresh = 0.95

    def log_face(self, uid):
        curr_time = datetime.now().timestamp()
        print("[FaceMatcher]", f"  Attempting to log face for user_id: {uid}")
        if uid not in self.last_log:
            self.last_log[uid] = curr_time
            print("[FaceMatcher]", f"  First log for user_id {uid}")
            return True

        # Only log if 5 minutes (300 seconds) have passed since last log
        time_diff = curr_time - self.last_log[uid]
        print(
            "[FaceMatcher]",
            f"  Time since last log for user_id {uid}: {time_diff:.2f} seconds",
        )
        if time_diff >= 300:
            self.last_log[uid] = curr_time
            print(
                "[FaceMatcher]",
                f"  Logging face for user_id {uid} after {time_diff:.2f} seconds",
            )
            return True
        print(
            "[FaceMatcher]",
            f"  Skipping log for user_id {uid} - too soon since last log",
        )
        return False

    def save_face(self, face_img, uid, is_known=True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(
            "[FaceMatcher]",
            f"  Saving face image for user_id: {uid}, is_known: {is_known}",
        )
        if is_known:
            save_dir = known / str(uid)
        else:
            save_dir = unknown
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"face_{ts}.jpg"
        print("[FaceMatcher]", f"  Saving face image to: {save_path}")
        cv.imwrite(str(save_path), face_img)
        return str(save_path)

    def update_hist(self, face_idx, uid, conf, bbox):
        if face_idx not in self.rec_hist:
            self.rec_hist[face_idx] = []
            self.pos_hist[face_idx] = []
            self.id_stab[face_idx] = 0.0

        self.rec_hist[face_idx].append((uid, conf))
        self.pos_hist[face_idx].append(bbox)

        if len(self.rec_hist[face_idx]) > self.hist_win:
            self.rec_hist[face_idx].pop(0)
            self.pos_hist[face_idx].pop(0)

    def calc_stab(self, face_idx, curr_uid, curr_conf):
        if face_idx not in self.rec_hist:
            return 0.0

        hist = self.rec_hist[face_idx]
        if not hist:
            return 0.0

        same_id_count = sum(1 for uid, _ in hist if uid == curr_uid)
        stab = same_id_count / len(hist)

        self.id_stab[face_idx] = stab * curr_conf
        return self.id_stab[face_idx]

    def get_consistent_id(self, face_idx, curr_uid, curr_conf, bbox):
        if face_idx not in self.rec_hist:
            return curr_uid, curr_conf

        hist = self.rec_hist[face_idx]
        pos = self.pos_hist[face_idx]

        if not hist:
            return curr_uid, curr_conf

        curr_stab = self.calc_stab(face_idx, curr_uid, curr_conf)

        if face_idx in self.last_ids:
            last_id = self.last_ids[face_idx]
            last_conf = self.id_conf.get(face_idx, 0)

            if last_id in [uid for uid, _ in hist]:
                last_stab = self.calc_stab(face_idx, last_id, last_conf)

                if last_stab > curr_stab and last_stab > 0.5:
                    return last_id, last_conf

        return curr_uid, curr_conf

    def is_ambiguous(self, max_sim, nxt_sim, ratio, margin, is_high_sim):
        if is_high_sim:
            if max_sim >= 0.99:
                if nxt_sim >= 0.99:
                    return False
                if margin < 0.001 and ratio >= 1.001:
                    return False
                return margin < 0.005 and ratio < 1.005
            else:
                if nxt_sim >= 0.98:
                    return False
                if margin < 0.005 and ratio >= 1.005:
                    return False
                return margin < 0.01 and ratio < 1.01
        else:
            return margin < 0.02 and ratio < 1.1

    def match_faces(self, embeds, boxes, imgs):
        print("[FaceMatcher]", "  Starting face matching process")
        db_rows = db.get_face_embeds()
        print(
            "[FaceMatcher]",
            f"  Retrieved {len(db_rows) if db_rows else 0} face embeddings from database",
        )
        db_embeds, db_info = [], []

        for row in db_rows:
            db_embed = np.frombuffer(row["face_embedding"], dtype=np.float32)
            db_embeds.append(db_embed)
            db_info.append(
                {
                    "user_idx": row["user_idx"],
                    "firstname": row["firstname"],
                    "lastname": row["lastname"],
                    "role": row["role"],
                }
            )

        db_embeds = np.stack(db_embeds) if db_embeds else np.zeros((0, 512))
        results = []

        if len(embeds) > 0 and len(db_embeds) > 0:
            print(
                "[FaceMatcher]",
                f"  Matching {len(embeds)} detected faces against {len(db_embeds)} database faces",
            )

            embeds = np.stack(embeds)
            sim_mat = np.dot(embeds, db_embeds.T)  # Calculate cosine similarity matrix
            row_ind, col_ind = linear_sum_assignment(-sim_mat)
            assigned_db = set()

            HIGH_SIM = 0.985
            LOW_SIM = 0.90

            # Process each face
            for i, j in zip(row_ind, col_ind):
                sims = np.sort(sim_mat[i])[::-1]
                max_sim = sims[0]
                nxt_sim = sims[1] if len(sims) > 1 else 0.0
                ratio = max_sim / (nxt_sim + 1e-6)
                margin = max_sim - nxt_sim
                is_high_sim = max_sim >= HIGH_SIM
                ambiguous = self.is_ambiguous(
                    max_sim, nxt_sim, ratio, margin, is_high_sim
                )

                print("[FaceMatcher]", f"Face {i} matching details:")
                print("[FaceMatcher]", f"Best similarity: {max_sim:.4f}")
                print("[FaceMatcher]", f"Second best similarity: {nxt_sim:.4f}")
                print("[FaceMatcher]", f"Ratio: {ratio:.4f}")
                print("[FaceMatcher]", f"Margin: {margin:.4f}")
                print("[FaceMatcher]", f"Ambiguous: {ambiguous}")
                print("[FaceMatcher]", f"High similarity: {is_high_sim}")

                if len(db_embeds) == 1:
                    if max_sim < LOW_SIM or j in assigned_db:
                        identity = "unknown"
                        conf = 0.0
                        role = ""
                        uid = -1
                        print(
                            "[FaceMatcher]",
                            f"Face {i} rejected - low similarity or already assigned",
                        )
                    else:
                        identity = f"{db_info[j]['firstname']} {db_info[j]['lastname']}"
                        conf = float(max_sim)
                        role = db_info[j]["role"]
                        uid = db_info[j]["user_idx"]
                        assigned_db.add(j)
                        print(
                            "[FaceMatcher]",
                            f"Face {i} matched to {identity} with confidence {conf:.4f}",
                        )
                else:
                    if (
                        max_sim < LOW_SIM
                        or ambiguous
                        or j in assigned_db
                        or (max_sim < HIGH_SIM and margin < 0.02)
                    ):
                        identity = "unknown"
                        conf = 0.0
                        role = ""
                        uid = -1
                        print(
                            "[FaceMatcher]",
                            f"Face {i} rejected - low similarity, ambiguous, or already assigned",
                        )
                    else:
                        identity = f"{db_info[j]['firstname']} {db_info[j]['lastname']}"
                        conf = float(max_sim)
                        role = db_info[j]["role"]
                        uid = db_info[j]["user_idx"]
                        assigned_db.add(j)
                        print(
                            "[FaceMatcher]",
                            f"Face {i} matched to {identity} with confidence {conf:.4f}",
                        )

                x1, y1, x2, y2 = boxes[i]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                uid, conf = self.get_consistent_id(i, uid, conf, bbox)
                self.update_hist(i, uid, conf, bbox)
                self.last_ids[i] = uid
                self.id_conf[i] = conf

                if self.log_face(uid):
                    print("[FaceMatcher]", f"  Logging face for user_id {uid}")
                    recog_img_path = self.save_face(imgs[i], uid, is_known=(uid != -1))
                    print(
                        "[FaceMatcher]", f"Inserting recognition log for user_id {uid}"
                    )
                    db.insert_recog_log(uid, conf, recog_img_path, identity, "face")
                    print("[FaceMatcher]", f"  Recognition log inserted successfully")
                else:
                    print(
                        "[FaceMatcher]", f"Skipping recognition log for user_id {uid}"
                    )

                results.append(
                    {
                        "bbox": bbox,
                        "identity": identity,
                        "confidence": conf,
                        "role": role,
                    }
                )

            for i in range(len(embeds)):
                if i not in row_ind:
                    x1, y1, x2, y2 = boxes[i]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    results.append(self.proc.unk_face_result(bbox))
                    print("[FaceMatcher]", f"Face {i} not matched to any database face")

        else:
            print("[FaceMatcher]", "No faces detected or no database faces available")
            for i in range(len(embeds)):
                x1, y1, x2, y2 = boxes[i]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                results.append(self.proc.unk_face_result(bbox))

        print("[FaceMatcher]", f"Face matching completed. Found {len(results)} results")
        return results
