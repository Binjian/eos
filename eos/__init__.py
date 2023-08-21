from pathlib import Path

from .data_io.config.vcu_calib_generator import generate_vcu_calibration

proj_root = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()


__all__ = [
    'generate_vcu_calibration',
    'proj_root',
]
