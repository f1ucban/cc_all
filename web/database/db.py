import sqlite3
from flask import g
from contextlib import contextmanager

from web.utils.config import database


def conn_db():
    if "db" not in g:
        try:
            g.db = sqlite3.connect(str(database), check_same_thread=False)
            g.db.row_factory = sqlite3.Row
        except Exception as e:
            print("[Database]", f"Database connection error: {e}")
            raise
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        try:
            db.close()
        except Exception as e:
            print("[Database]", f"Error closing database: {e}")


@contextmanager
def get_cursor():
    db = conn_db()
    cursor = db.cursor()
    try:
        yield cursor
        db.commit()
    except Exception as e:
        db.rollback()
        print("[Database]", f"Database error: {e}")
        raise


def exec_query(query, params=(), fetch_all=False):
    try:
        with get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch_all:
                return cursor.fetchall()
            return cursor.fetchone()
    except Exception as e:
        print("[Database]", f"Query execution error: {e}")
        raise


def init_db():
    try:
        with get_cursor() as cursor:
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    firstname TEXT NOT NULL,
                    lastname TEXT NOT NULL,
                    role TEXT,
                    date_enrolled TEXT DEFAULT CURRENT_TIMESTAMP,
                    profile_img TEXT,
                    UNIQUE(firstname, lastname)
                );

                CREATE TABLE IF NOT EXISTS face_data (
                    face_idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_idx INTEGER,
                    face_embedding BLOB NOT NULL,
                    date_enrolled TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_idx) REFERENCES users(user_idx) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS gait_enrollments (
                    enrollment_idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_idx INTEGER NOT NULL,
                    gait_features BLOB NOT NULL,
                    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_idx) REFERENCES users(user_idx) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS fused_data (
                    fused_idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_idx INTEGER NOT NULL,
                    fused_features BLOB NOT NULL,
                    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_idx) REFERENCES users(user_idx) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recog_logs (
                    recog_idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_idx INTEGER,
                    confidence REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    recog_img TEXT,
                    recog_result TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    FOREIGN KEY(user_idx) REFERENCES users(user_idx) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_recog_user ON recog_logs(user_idx);
                CREATE INDEX IF NOT EXISTS idx_recog_result ON recog_logs(recog_result);
                CREATE INDEX IF NOT EXISTS idx_recog_modality ON recog_logs(modality);
                CREATE INDEX IF NOT EXISTS idx_fused_user ON fused_data(user_idx);
                """
            )
    except Exception as e:
        print("[Database]", f"Database initialization error: {e}")
        raise


def get_all_users():
    return exec_query(
        """
        SELECT u.user_idx, u.firstname, u.lastname, u.role, u.date_enrolled, 
               u.profile_img,
               f.face_embedding IS NOT NULL as has_face,
               g.gait_features IS NOT NULL as has_gait,
               g.gait_features
        FROM users u
        LEFT JOIN face_data f ON u.user_idx = f.user_idx
        LEFT JOIN gait_enrollments g ON u.user_idx = g.user_idx
        GROUP BY u.user_idx
        ORDER BY u.lastname, u.firstname
        """,
        fetch_all=True,
    )


def user_pfp(user_idx):
    return exec_query("SELECT profile_img FROM users WHERE user_idx = ?", (user_idx,))


def delete_user(user_idx):
    exec_query("DELETE FROM users WHERE user_idx = ?", (user_idx,))


def insert_user(firstname, lastname, role, profile_img=None):
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
        INSERT INTO users (firstname, lastname, role, profile_img) 
        VALUES (?, ?, ?, ?)
        RETURNING user_idx
        """,
                (firstname, lastname, role, profile_img),
            )
            result = cursor.fetchone()
            return result["user_idx"] if result else None
    except Exception as e:
        print("[Database]", f"Error inserting user: {str(e)}")
        return None


def insert_face_data(user_idx, face_embedding):
    exec_query(
        "INSERT INTO face_data (user_idx, face_embedding) VALUES (?, ?)",
        (user_idx, face_embedding),
    )


def insert_gait_enrollment(user_idx, gait_features):
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO gait_enrollments (user_idx, gait_features) 
                VALUES (?, ?)
                RETURNING enrollment_idx
                """,
                (user_idx, gait_features),
            )
            result = cursor.fetchone()
            return result["enrollment_idx"] if result else None
    except Exception as e:
        print("[Database]", f"Error inserting gait enrollment: {str(e)}")
        return None


def get_user_gait_enrollments(user_idx):
    try:
        return exec_query(
            """
            SELECT enrollment_idx, gait_features, enrollment_date 
            FROM gait_enrollments 
            WHERE user_idx = ?
            ORDER BY enrollment_date DESC
            """,
            (user_idx,),
            fetch_all=True,
        )
    except Exception as e:
        print("[Database]", f"Error getting gait enrollments: {str(e)}")
        return []


def delete_gait_enrollment(enrollment_idx):
    try:
        exec_query(
            "DELETE FROM gait_enrollments WHERE enrollment_idx = ?", (enrollment_idx,)
        )
        return True
    except Exception as e:
        print("[Database]", f"Error deleting gait enrollment: {str(e)}")
        return False


def get_gait_features():
    return exec_query(
        """
        SELECT u.user_idx, u.firstname, u.lastname, u.role, 
               g.enrollment_idx, g.gait_features, g.enrollment_date
        FROM users u
        JOIN gait_enrollments g ON u.user_idx = g.user_idx
        ORDER BY u.lastname, u.firstname, g.enrollment_date DESC
        """,
        fetch_all=True,
    )


def get_face_embeds():
    return exec_query(
        """
        SELECT u.user_idx, u.firstname, u.lastname, u.role, f.face_embedding 
        FROM users u
        JOIN face_data f ON u.user_idx = f.user_idx
        """,
        fetch_all=True,
    )


def user_exists(firstname, lastname):
    return exec_query(
        "SELECT user_idx FROM users WHERE firstname = ? AND lastname = ?",
        (firstname, lastname),
    )


def get_or_create_user(firstname, lastname, role=None):
    user = user_exists(firstname, lastname)
    if user:
        return user["user_idx"]
    return insert_user(firstname, lastname, role)["user_idx"]


def insert_recog_log(user_idx, confidence, recog_img, recog_result, modality="face"):
    print("[Database]", f"Inserting recognition log:")
    print("[Database]", f"- user_idx: {user_idx}")
    print("[Database]", f"- confidence: {confidence}")
    print("[Database]", f"- recog_img: {recog_img}")
    print("[Database]", f"- recog_result: {recog_result}")
    print("[Database]", f"- modality: {modality}")
    try:
        exec_query(
            """
            INSERT INTO recog_logs (user_idx, confidence, recog_img, recog_result, modality) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_idx, confidence, recog_img, recog_result, modality),
        )
        print("[Database]", "Recognition log inserted successfully")
    except Exception as e:
        print("[Database]", f"Error inserting recognition log: {str(e)}")
        raise


def get_recognition_logs():
    print("[Database]", "Retrieving recognition logs")
    try:
        logs = exec_query(
            """
            SELECT 
                r.recog_idx,
                r.confidence,
                r.timestamp,
                r.recog_img,
                r.recog_result as identity,
                r.modality,
                u.firstname,
                u.lastname,
                u.role
            FROM recog_logs r
            LEFT JOIN users u ON r.user_idx = u.user_idx
            ORDER BY r.timestamp DESC
            LIMIT 100
            """,
            fetch_all=True,
        )
        print("[Database]", f"Retrieved {len(logs) if logs else 0} recognition logs")
        return logs
    except Exception as e:
        print("[Database]", f"Error retrieving recognition logs: {str(e)}")
        raise


def update_gait_features(user_idx, features_bytes):
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT user_idx FROM users WHERE user_idx = ?", (user_idx,))
            if not cursor.fetchone():
                return False

            cursor.execute(
                "SELECT enrollment_idx FROM gait_enrollments WHERE user_idx = ?",
                (user_idx,),
            )
            existing_gait = cursor.fetchone()

            if existing_gait:
                cursor.execute(
                    "UPDATE gait_enrollments SET gait_features = ? WHERE user_idx = ?",
                    (features_bytes, user_idx),
                )
            else:
                cursor.execute(
                    "INSERT INTO gait_enrollments (user_idx, gait_features) VALUES (?, ?)",
                    (user_idx, features_bytes),
                )
            return True
    except Exception as e:
        print("[Database]", f"Error updating gait features: {str(e)}")
        return False


def update_user_profile(user_idx, profile_img, role=None):
    try:
        with get_cursor() as cursor:
            if role is not None:
                cursor.execute(
                    """
                    UPDATE users 
                    SET profile_img = ?, role = ?
                    WHERE user_idx = ?
                    """,
                    (profile_img, role, user_idx),
                )
            else:
                cursor.execute(
                    """
                    UPDATE users 
                    SET profile_img = ?
                    WHERE user_idx = ?
                    """,
                    (profile_img, user_idx),
                )
            return True
    except Exception as e:
        print("[Database]", f"Error updating user profile: {str(e)}")
        return False


def get_user_by_id(user_idx):
    return exec_query(
        "SELECT * FROM users WHERE user_idx = ?", (user_idx,), fetch_all=False
    )


def get_user_by_name(firstname, lastname):
    return exec_query(
        "SELECT user_idx FROM users WHERE firstname = ? AND lastname = ?",
        (firstname, lastname),
    )


def insert_fused_data(user_idx, fused_features):
    try:
        with get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO fused_data (user_idx, fused_features) 
                VALUES (?, ?)
                RETURNING fused_idx
                """,
                (user_idx, fused_features),
            )
            result = cursor.fetchone()
            return result["fused_idx"] if result else None
    except Exception as e:
        print("[Database]", f"Error inserting fused data: {str(e)}")
        return None


def get_fused_features():
    return exec_query(
        """
        SELECT u.user_idx, u.firstname, u.lastname, u.role, f.fused_features 
        FROM users u
        JOIN fused_data f ON u.user_idx = f.user_idx
        """,
        fetch_all=True,
    )


def update_fused_features(user_idx, features_bytes):
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT user_idx FROM users WHERE user_idx = ?", (user_idx,))
            if not cursor.fetchone():
                return False

            cursor.execute(
                "SELECT fused_idx FROM fused_data WHERE user_idx = ?",
                (user_idx,),
            )
            existing_fused = cursor.fetchone()

            if existing_fused:
                cursor.execute(
                    "UPDATE fused_data SET fused_features = ? WHERE user_idx = ?",
                    (features_bytes, user_idx),
                )
            else:
                cursor.execute(
                    "INSERT INTO fused_data (user_idx, fused_features) VALUES (?, ?)",
                    (user_idx, features_bytes),
                )
            return True
    except Exception as e:
        print("[Database]", f"Error updating fused features: {str(e)}")
        return False


def get_user_fused_features(user_idx):
    return exec_query(
        """
        SELECT fused_features 
        FROM fused_data 
        WHERE user_idx = ?
        ORDER BY enrollment_date DESC
        LIMIT 1
        """,
        (user_idx,),
    )
