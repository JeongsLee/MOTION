"""motion_torch: PyTorch port of the MOTION PDE model (motion_tf v3, TensorFlow) for weight loading, inference and
fine-tuning. Pretraining stays in TensorFlow. Does not import TensorFlow."""
from .config import PRESETS, MotionConfig  # noqa: F401
from .model.v3model import V3Model, interp_box, to_grid  # noqa: F401
from .convert import build_model, load_tf_npz  # noqa: F401
