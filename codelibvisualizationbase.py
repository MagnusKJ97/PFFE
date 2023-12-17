from typing import NoReturn, Union, List

import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codelibvisualizationlayout import default_colors
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
import seaborn as sns

def fan_chart(x: ndarray, y: ndarray, **kwargs) -> NoReturn:

    """
    Plots a fan chart.

    If the number of rows of `y` is divisible by 2, the middle row of `y` is plotted as a line in the middle

    Parameters
    ----------
    x: ndarray
        Vector representing the "x-values" of the plot
    y: ndarray
        Matrix of data to plot. Number of columns equal to the length of `x`. Number of rows / 2 is equal to the number
        different colored areas in the plot. It is assumed that values in the first row is smaller than the values in the
        second row and so on.
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

            import numpy as np
            from codelib.visualization.base import fan_chart
            data = np.array([np.random.normal(size=1000) * s for s in np.arange(0, 1, 0.1)])
            percentiles = np.percentile(data, [10, 20, 50, 80, 90], axis=1)
            fan_chart(np.arange(1, 11, 1), percentiles, labels=['80% CI', '60% CI', 'median'])
            plt.show()

    """

    # defaults
    color_perc = "blue"
    color_median = "red"
    xlabel = None
    ylabel = None
    title = None
    labels = None
    initialize_fig = True

    if 'color' in kwargs:
        color_perc = kwargs['color']
    if 'color_median' in kwargs:
        color_median = kwargs['color_median']
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'labels' in kwargs:
        labels = True
        labels_to_plot = kwargs['labels']
    if "fig" in kwargs:
        fig = kwargs["fig"]
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False

    number_of_rows = y.shape[0]
    number_to_plot = number_of_rows // 2

    if labels is None:
        labels_to_plot = ["" for i in range(number_to_plot + number_of_rows % 2)]

    if initialize_fig:
        fig, ax = plt.subplots()

    for i in range(number_to_plot):

        # for plotting below
        values1 = y[i, :]
        values2 = y[i + 1, :]

        # for plotting above
        values3 = y[-2 - i, :]
        values4 = y[-1 - i, :]

        # calculate alpha
        alpha = 0.95 * (i + 1) / number_to_plot

        ax.fill_between(x, values1, values2, alpha=alpha, color=color_perc, label=labels_to_plot[i])
        ax.fill_between(x, values3, values4, alpha=alpha, color=color_perc)

    # plot center value with specific color
    if number_of_rows % 2 == 1:
        ax.plot(x, y[number_to_plot], color=color_median, label=labels_to_plot[-1])

    # add title
    plt.title(title)
    # add label to x axis
    plt.xlabel(xlabel)
    # add label to y axis
    plt.ylabel(ylabel)
    # legend
    if labels:
        ax.legend()



def correlation_plot(correlation_matrix: np.ndarray, names: Union[List[str], None] = None, **kwargs) -> None:

    """
    Plots a correlation matrix using seaborn heatmap

    Parameters
    ----------
    correlation_matrix: ndarray
        Matrix with entries being correlations
    names: List
        List of names representing the variables names. Ordering is the same as for the correlation matrix
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from corelib.plotting import correlation_plot
        data = np.random.normal(size=(10, 1000))
        corr = np.corrcoef(data)
        correlation_plot(corr)
        plt.show()

    """
    # my_params = mpl.rcParams
    # sns.set(style="white")

    mask_upper_diagonal = True
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmax = 1.0
    vmin = -1.0
    center = 0.0
    cbar_kws = {"shrink": .75}
    title = None
    include_diagonal = False
    include_values = False
    fmt = "d"
    size_scale = False

    if 'mask' in kwargs:
        mask_upper_diagonal = kwargs['mask']
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    if 'center' in kwargs:
        center = kwargs['center']
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'cbar_kws' in kwargs:
        cbar_kws = kwargs['cbar_kws']
    if 'include_diagonal' in kwargs:
        include_diagonal = kwargs['include_diagonal']
    if 'include_values' in kwargs:
        include_values = kwargs['include_values']
    if 'size_scale' in kwargs:
        size_scale = kwargs['size_scale']

    # create data frame with correlation matrix
    df_correlation = pd.DataFrame(correlation_matrix, columns=names, index=names)

    mask = None
    if mask_upper_diagonal:
        k = 0
        if include_diagonal:
            k = 1
        mask = np.triu(np.ones_like(df_correlation, dtype=bool), k=k)

#    fig, ax = plt.subplots()

    sns.heatmap(df_correlation, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                square=True, linewidths=.5, cbar_kws=cbar_kws, annot=include_values)

    # add title
    plt.title(title)

    # restore default values
    # mpl.rcParams.update(my_params)
