from pathlib import Path
from flask import request
from datetime import datetime


from web.utils.config import enrollment
from web.utils.helpers import (
    error_response,
    success_response,
    create_user_dirs,
    validate_user_input,
    get_user_data,
)


class UserController:
    def __init__(self, db):
        self.db = db

    def create_user(self):
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
        user_path = (
            enrollment / f"{user_data['firstname']}_{user_data['lastname']}_{timestamp}"
        )

        success, error = create_user_dirs(user_path, timestamp)
        if not success:
            return error_response(error)

        user_idx = self.db.insert_user(
            user_data["firstname"], user_data["lastname"], user_data["role"]
        )
        if not user_idx:
            return error_response("Failed to create user")

        return success_response({"user": {"user_idx": user_idx, **user_data}})

    def remove_user(self, user_idx):
        row = self.db.user_pfp(user_idx)
        if not row:
            return error_response("User not found", 404)

        pfp = Path(row["profile_img"]) if row["profile_img"] else None
        self.db.delete_user(user_idx)

        if pfp and pfp.exists():
            pfp.unlink()

        return success_response()

    def list_users(self):
        users = self.db.get_all_users()
        if users is None:
            return error_response("Failed to retrieve users")

        user_list = []
        for row in users:
            user_dict = dict(row)
            user_dict.pop("face_embedding", None)
            user_dict.pop("gait_features", None)
            user_list.append(user_dict)

        return success_response({"users": user_list})
