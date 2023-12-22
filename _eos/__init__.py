__version__ = "0.0.1"
from pathlib import Path

import tensorflow as tf

proj_root = Path(__file__).parent.parent


# gpus = tf.config.list_physical_devices(device_type="GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

__all__ = [
    "proj_root",
]
