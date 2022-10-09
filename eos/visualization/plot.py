# system import

# third party import
import matplotlib.pyplot as plt
import pandas as pd

# local import


def plot_3d_figure(table: pd.DataFrame):
    """Create a matplotlib 3d figure, //export and save in log
    table: pd.DataFrame
    """

    df = table.unstack().reset_index()
    df.columns = ["pedal", "velocity", "torque"]

    fig = plt.figure(visible=False)
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(
        df["pedal"],
        df["velocity"],
        df["torque"],
        cmap=plt.cm.YlGnBu,
        linewidth=5,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(30, 175)

    return fig
    # plt.show()
    # time.sleep(5)
