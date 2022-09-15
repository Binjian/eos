# system import

# third party import
import matplotlib.pyplot as plt
import pandas as pd

# local import


def plot_3d_figure(table, pd_columns, pd_index):
    """Create a matplotlib 3d figure, //export and save in log"""
    pd_data = pd.DataFrame(
        table,
        columns=pd_columns,
        index=pd_index,
    )
    df = pd_data.unstack().reset_index()
    df.columns = ["pedal", "velocity", "throttle"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(
        df["pedal"],
        df["velocity"],
        df["throttle"],
        cmap=plt.cm.viridis,
        linewidth=0.2,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(30, 135)

    return fig
    # plt.show()
    # time.sleep(5)
