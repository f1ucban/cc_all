import torch
import shutil
import traceback
from pathlib import Path
from flask import jsonify
from functools import wraps
from easydict import EasyDict
from werkzeug.utils import secure_filename
from web.utils.config import device, finetuned
from src.face.model.arcface_model import ArcFaceModel


class ModelLoader:
    @staticmethod
    def load_model():
        try:
            torch.serialization.add_safe_globals([EasyDict])
            ckpt = torch.load(finetuned, map_location=device, weights_only=False)
            model = ArcFaceModel(n_cls=60)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device).eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
    def init_gpu():
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        return gpu_available


class ResponseBuilder:
    @staticmethod
    def error(error_msg, status_code=500, include_traceback=False):
        response = {"error": str(error_msg)}
        if include_traceback:
            response["details"] = traceback.format_exc()
        return jsonify(response), status_code

    @staticmethod
    def success(data=None, message=None):
        response = {"status": "success"}
        if data:
            response.update(data)
        if message:
            response["message"] = message
        return jsonify(response)


class UserValidator:
    @staticmethod
    def validate(firstname, lastname):
        if not firstname or not lastname:
            return False, "Provide full name"
        return True, None

    @staticmethod
    def get_data(firstname, lastname, role=""):
        return {
            "firstname": secure_filename(firstname),
            "lastname": secure_filename(lastname),
            "role": secure_filename(role),
        }


class FileManager:
    @staticmethod
    def create_user_dirs(base_path, timestamp, enrollment_type="face"):
        try:
            base_path = Path(base_path).resolve()
            base_path.mkdir(parents=True, exist_ok=True)
            timestamp_path = base_path / f"{enrollment_type}_{timestamp}"
            timestamp_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directories: {base_path} and {timestamp_path}")
            return True, None
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            return False, str(e)

    @staticmethod
    def cleanup_dir(path):
        try:
            if path.exists():
                shutil.rmtree(str(path))
            return True
        except Exception:
            return False


def handle_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return ResponseBuilder.error(e, include_traceback=True)

    return wrapper


load_model = ModelLoader.load_model
init_gpu = ModelLoader.init_gpu
error_response = ResponseBuilder.error
success_response = ResponseBuilder.success
validate_user_input = UserValidator.validate
get_user_data = UserValidator.get_data
create_user_dirs = FileManager.create_user_dirs
cleanup_dir = FileManager.cleanup_dir
