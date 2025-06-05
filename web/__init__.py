import os
from flask import Flask
from flask_cors import CORS


from web.database import db
from web.utils.helpers import load_model
from web.utils.config import uploads, device, database
from web.utils.convert import convert_arcface_to_onnx, create_onnx_session


from web.face.f_match import FaceMatcher
from web.face.f_proc import FaceProcessor


from web.gait.pose import PoseProcessor
from web.gait.g_proc import GaitProcessor

from web.controllers.user_ctrl import UserController
from web.controllers.face_ctrl import FaceController
from web.controllers.gait_ctrl import GaitController
from web.controllers.pose_ctrl import PoseController


def init_app(use_onnx=False):
    app = Flask(__name__)
    CORS(app)

    app.config["UPLOADS"] = str(uploads)
    app.config["MAX_CONTENT"] = 16 * 1024 * 1024
    os.makedirs(app.config["UPLOADS"], exist_ok=True)
    os.makedirs(os.path.dirname(database), exist_ok=True)

    if use_onnx:
        onnx_path = convert_arcface_to_onnx()
        arcface = create_onnx_session(onnx_path)
    else:
        arcface = load_model()

    face_proc = FaceProcessor(device, arcface, use_onnx=use_onnx)
    face_match = FaceMatcher(face_proc)
    pose_proc = PoseProcessor()
    gait_proc = GaitProcessor()

    user_controller = UserController(db)
    face_controller = FaceController(face_proc, face_match)
    pose_controller = PoseController(pose_proc)
    gait_controller = GaitController(gait_proc, pose_controller)

    app.teardown_appcontext(db.close_db)

    with app.app_context():
        try:
            db.init_db()
        except Exception as e:
            print(f"Error initializing DB: {e}")
            raise

    return (app, user_controller, face_controller, pose_controller, gait_controller)
