from pathlib import Path

import tensorflow as tf

from .data_io.config.vcu_calib_generator import generate_vcu_calibration

proj_root = Path(__file__).parent.parent


gpus = tf.config.list_physical_devices(device_type="GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

# TODO: Add logging support

# tracer = VizTracer()


__all__ = [
    "generate_vcu_calibration",
    "proj_root",
]
