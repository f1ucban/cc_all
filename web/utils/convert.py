import torch
import onnx
from pathlib import Path
import onnxruntime as ort
from web.utils.config import device, finetuned
from web.utils.helpers import ModelLoader


class ONNXConverter:
    def __init__(self, model_path=None, output_path=None):
        self.model_path = model_path or finetuned
        self.output_path = (
            output_path or Path(self.model_path).parent / "arcface_model.onnx"
        )

    def convert(self):
        print(f"Loading PyTorch model from {self.model_path}")
        model = ModelLoader.load_model()
        self._export_to_onnx(model)
        self._verify_model()
        print(f"ONNX model saved to {self.output_path}")
        return self.output_path

    def _export_to_onnx(self, model):
        print(f"Converting model to ONNX format...")
        dummy_input = torch.randn(1, 3, 112, 112, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            self.output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def _verify_model(self):
        print("Verifying ONNX model...")
        onnx_model = onnx.load(self.output_path)
        onnx.checker.check_model(onnx_model)


class ONNXSession:
    def __init__(self, model_path_or_session):
        if isinstance(model_path_or_session, (str, Path)):
            self.session = self._create_session(str(model_path_or_session))
        elif isinstance(model_path_or_session, ort.InferenceSession):
            self.session = model_path_or_session
        else:
            raise TypeError(
                f"Expected string, Path, or InferenceSession, got {type(model_path_or_session)}"
            )

    def _create_session(self, model_path):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        return ort.InferenceSession(
            model_path,
            sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def run_inference(self, input_tensor):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        return self.session.run([output_name], {input_name: input_tensor})[0]


def convert_arcface_to_onnx(model_path=None, output_path=None):
    converter = ONNXConverter(model_path, output_path)
    return converter.convert()


def create_onnx_session(model_path):
    return ONNXSession(model_path).session


def run_onnx_inference(session, input_tensor):
    return ONNXSession(session).run_inference(input_tensor)


if __name__ == "__main__":
    onnx_path = convert_arcface_to_onnx()
    session = create_onnx_session(onnx_path)
    dummy_input = torch.randn(1, 3, 112, 112).numpy()
    output = run_onnx_inference(session, dummy_input)
    print(f"Test inference successful. Output shape: {output.shape}")
