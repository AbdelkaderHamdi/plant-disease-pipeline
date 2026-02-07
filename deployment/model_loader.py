import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")


def list_models(with_segmentation=False):
    folder = "with_segmentation" if with_segmentation else "without_segmentation"
    model_dir = os.path.join(MODELS_DIR, folder)

    if not os.path.exists(model_dir):
        return []

    # model files only (exclude label encoders and metrics)
    return [
        f for f in os.listdir(model_dir)
        if f.endswith(".joblib")
        and "_le_" not in f
    ]


def load_model(model_name, with_segmentation=False):
    folder = "with_segmentation" if with_segmentation else "without_segmentation"
    model_dir = os.path.join(MODELS_DIR, folder)

    model_path = os.path.join(model_dir, model_name)

    # build correct label encoder name
    # Apple_random_forest_20260207_151513.joblib
    # -> Apple_random_forest_le_20260207_151513.joblib
    parts = model_name.replace(".joblib", "").rsplit("_", 2)
    le_name = f"{parts[0]}_le_{parts[1]}_{parts[2]}.joblib"
    le_path = os.path.join(model_dir, le_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_name}")

    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found: {le_name}")

    model = joblib.load(model_path)
    le = joblib.load(le_path)

    return model, le


def latest_model(with_segmentation=False):
    models = list_models(with_segmentation)
    if not models:
        return None, None

    # filenames already contain sortable timestamps
    latest = sorted(models, reverse=True)[0]
    return load_model(latest, with_segmentation)
