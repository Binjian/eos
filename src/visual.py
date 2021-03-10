import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


def visual(t_real, v_real, v_wltc):
    plt.plot(t_real, v_real, color="green")
    plt.plot(t_real, v_wltc, color="blue")
    plt.legend(["Actual Velocity", "WLTC Velocity"])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/h)")
    plt.title("Comparison between actual V and WLTC V")
    plt.pause(0.0001)
