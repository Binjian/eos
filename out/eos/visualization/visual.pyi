from matplotlib import style as style
from reportlab.platypus import Spacer as Spacer, Table as Table

def plot_to_image(figure): ...
def visual(t_real, v_real, v_wltc) -> None: ...
def compare_pic(t_real, t, e, e_real, thro_dev, thro_real_dev) -> None: ...
def gen_report(diff, diff1, e) -> None: ...
