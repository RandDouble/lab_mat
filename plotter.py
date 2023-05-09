"""
Seeing that the operation to plotting are the same, and datas to be analyzed will increase, it can be automatized
More keystroke now is less keystroke tomorrow
"""

import matplotlib.pyplot as plt
import pandas as pd


def plotter(
    data: pd.DataFrame,
    col_to_print: list,
    *,
    zero: pd.DataFrame = None,
    zero_columns: list = None,
    label: list | str = None,
    axis_names: list = (r"$lambda$ [m]", "Transmittance")
):
    """
    function to plot with some common denom:
        - data: principal data to print, it is a DataFrame
        - col_to_print: which column of data to print
        - zero: if there is a DataFrame of reference to print
        - zero_columns: which col of zero to print
        - label: its name is self explanatory
        - axis_name: same as above
    """

    fig, ax = plt.subplots()

    ax.plot(*col_to_print, data=data, label=label[0])
    if zero:
        ax.plot(*zero_columns, data=zero, label=label[1])
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.legend()

    return fig, ax
