import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = os.environ.get("YOLO_WEIGHTS") or os.path.join(HERE, "..", "weights", "yolo26n_person_phone.pt")

_model = None


def load(path=WEIGHTS):
    global _model
    if _model is None:
        _model = YOLO(path)
    return _model


if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    model.export(format="onnx")
