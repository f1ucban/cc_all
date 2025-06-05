from flask import render_template, request
from web import init_app
from web.utils.helpers import (
    init_gpu,
    handle_exception,
    error_response,
    success_response,
)


app, user_controller, face_controller, pose_controller, gait_controller = init_app(
    use_onnx=True
)
gpu_available = init_gpu()
LOGS_PASSWORD = "password123"


@app.route("/")
@handle_exception
def index():
    return render_template("index.html")


@app.route("/create_user", methods=["POST"])
@handle_exception
def create_user():
    return user_controller.create_user()


@app.route("/remove_user/<int:user_idx>", methods=["DELETE"])
@handle_exception
def remove_user(user_idx):
    return user_controller.remove_user(user_idx)


@app.route("/users", methods=["GET"])
@handle_exception
def list_users():
    return user_controller.list_users()


@app.route("/pose_detect", methods=["POST"])
@handle_exception
def detect_pose():
    return pose_controller.detect_pose()


@app.route("/enroll", methods=["POST"])
@handle_exception
def enroll_face():
    return face_controller.enroll_face()


@app.route("/enroll_gait", methods=["POST"])
@handle_exception
def enroll_gait():
    return gait_controller.enroll_gait()


@app.route("/clear_gait_features", methods=["POST"])
@handle_exception
def clear_gait_features():
    return gait_controller.clear_gait_features()


@app.route("/process_frame", methods=["POST"])
@handle_exception
def process_frame():
    return face_controller.process_frame()


@app.route("/get_logs_records", methods=["POST"])
@handle_exception
def get_logs_records():
    password = request.form.get("password")

    if password != LOGS_PASSWORD:
        return error_response("Incorrect password", 401)

    users_response = user_controller.list_users()
    if users_response.status_code != 200:
        return users_response

    users_data = users_response.get_json()
    logs_data = []

    return success_response({"users": users_data.get("users", []), "logs": logs_data})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
