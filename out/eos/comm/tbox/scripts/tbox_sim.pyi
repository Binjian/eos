import pandas as pd

from eos.utils import prepend_string_arg as prepend_string_arg

g_tbox_sim_path: str
g_input_json_path: str
g_output_json_path: str
g_download_script_diffon: str
g_download_script_diffoff: str

def set_tbox_sim_path(tbox_sim_path) -> None: ...
def float_to_hex(value): ...
def hex_to_float(value): ...
def float_array_to_buffer(float_array): ...
def parse_arg(): ...
def write_json(output_json_path, example_json_path, data) -> None: ...
def send_float_array(name: str, float_df: pd.DataFrame, sw_diff: bool = ...): ...