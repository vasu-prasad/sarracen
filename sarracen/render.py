"""
Provides several rendering functions which produce matplotlib plots of SPH data.
These functions act as interfaces to interpolation functions within interpolate.py.

These functions can be accessed directly, for example:
    render_2d(data, target)
Or, they can be accessed through a `SarracenDataFrame` object, for example:
    data.render_2d(target)
"""

from typing import Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from sarracen.interpolate import interpolate_2d_cross, interpolate_2d, interpolate_3d, interpolate_3d_cross
from sarracen.kernels import BaseKernel


def _snap(value: float):
    """ Snap a number to the nearest integer

    Return a number which is rounded to the nearest integer,
    with a 1e-4 absolute range of tolerance.

    Parameters
    ----------
    value: float
        The number to snap.

    Returns
    -------
    float: An integer (in float form) if a close integer is detected, otherwise return `value`.
    """
    if np.isclose(value, np.rint(value), atol=1e-4):
        return np.rint(value)
    else:
        return value


def _default_axes(data, x, y):
    # x & y columns default to the variables determined by the SarracenDataFrame.
    if x is None:
        x = data.xcol
    if y is None:
        y = data.ycol

    return x, y


def _default_bounds(data, x, y, x_min, x_max, y_min, y_max):
    # boundaries of the plot default to the maximum & minimum values of the data.
    if x_min is None:
        x_min = _snap(data.loc[:, x].min())
    if y_min is None:
        y_min = _snap(data.loc[:, y].min())
    if x_max is None:
        x_max = _snap(data.loc[:, x].max())
    if y_max is None:
        y_max = _snap(data.loc[:, y].max())

    return x_min, x_max, y_min, y_max


def render_2d(data: 'SarracenDataFrame',
              target: str,
              x: str = None,
              y: str = None,
              kernel: BaseKernel = None,
              x_pixels: int = None,
              y_pixels: int = None,
              x_min: float = None,
              x_max: float = None,
              y_min: float = None,
              y_max: float = None,
              cmap: Union[str, Colormap] = 'RdBu',
              cbar: bool = True,
              cbar_kws: dict = {},
              cbar_ax: Axes = None,
              ax: Axes = None,
              **kwargs) -> Axes:
    """ Render 2D particle data to a 2D grid, using SPH rendering of a target variable.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target variable. The contributions of all particles near the rendered area are summed and
    stored to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x: str, optional
        Column label of the x-directional axis. Defaults to the x-column detected in `data`.
    y: str, optional
        Column label of the y-directional axis. Defaults to the y-column detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of y_pixels which
        preserves the aspect ratio of the data.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of x_pixels which
        preserves the aspect ratio of the data.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space). Defaults to the lower bound
        of x detected in `data` snapped to the nearest integer.
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space). Defaults to the upper bound
        of x detected in `data` snapped to the nearest integer.
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space). Defaults to the lower bound
        of y detected in `data` snapped to the nearest integer.
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space). Defaults to the upper bound
        of y detected in `data` snapped to the nearest integer.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.

    Returns
    -------
    Figure
        The resulting matplotlib figure, containing the 2d render and
        a color bar indicating the magnitude of the target variable.
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region.
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_2d(data, target, x, y, kernel, x_pixels, y_pixels, x_min, x_max, y_min, y_max)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    graphic = ax.imshow(image, cmap=cmap, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        colorbar.ax.set_ylabel(target)

    return ax


def render_2d_cross(data: 'SarracenDataFrame',
                    target: str,
                    x: str = None,
                    y: str = None,
                    kernel: BaseKernel = None,
                    pixels: int = 512,
                    x1: float = None,
                    x2: float = None,
                    y1: float = None,
                    y2: float = None,
                    ax: Axes = None,
                    **kwargs) -> Axes:
    """ Render 2D particle data to a 1D line, using a 2D cross-section.

    Render the data within a SarracenDataFrame to a seaborn-generated line plot, by taking
    a 2D->1D cross section of a target variable. The contributions of all particles near the
    cross-section line are summed and stored in a 1D array.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x: str, optional
        Column label of the x-directional axis. Defaults to the x-column detected in `data`.
    y: str, optional
        Column label of the y-directional axis. Defaults to the y-column detected in `data`.
    kernel: BaseKernel
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    pixels: int, optional
        Number of points in the resulting line plot in the x-direction.
    x1: float, optional
        Starting x-coordinate of the line (in particle data space). Defaults to the lower bound
        of x detected in `data` snapped to the nearest integer.
    x2: float, optional
        Ending x-coordinate of the line (in particle data space). Defaults to the upper bound
        of x detected in `data` snapped to the nearest integer.
    y1: float, optional
        Starting y-coordinate of the line (in particle data space). Defaults to the lower bound
        of y detected in `data` snapped to the nearest integer.
    y2: float, optional
        Ending y-coordinate of the line (in particle data space). Defaults to the upper bound
        of y detected in `data` snapped to the nearest integer.

    Returns
    -------
    Figure
        The resulting matplotlib figure, containing the seaborn-generated
        line plot.
    Axes
        The resulting matplotlib axes, which contain the line plot.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x1`, `x2`, `y1`, and `y2` are all the same
        (indicating a zero-length cross-section).
    KeyError
        If `target`, `x`, `y`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    output = interpolate_2d_cross(data, target, x, y, kernel, pixels, x1, x2, y1, y2)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x1, x2, y1, y2 = _default_bounds(data, x, y, x1, x2, y1, y2)

    sns.lineplot(x=np.linspace(0, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), pixels), y=output, ax=ax, **kwargs)
    ax.margins(x=0, y=0)
    ax.set_xlabel(f'cross-section ({x}, {y})')
    ax.set_ylabel(target)

    return ax


def render_3d(data: 'SarracenDataFrame',
              target: str,
              x: str = None,
              y: str = None,
              kernel: BaseKernel = None,
              integral_samples: int = 1000,
              rotation: np.ndarray = None,
              origin: np.ndarray = None,
              x_pixels: int = None,
              y_pixels: int = None,
              x_min: float = None,
              x_max: float = None,
              y_min: float = None,
              y_max: float = None,
              cmap: Union[str, Colormap] = 'RdBu',
              cbar: bool = True,
              cbar_kws: dict = {},
              cbar_ax: Axes = None,
              ax: Axes = None,
              **kwargs) -> Axes:
    """ Render 3D particle data to a 2D grid, using SPH column rendering of a target variable.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, by rendering the values
    of a target variable. The contributions of all particles near columns across the z-axis are
    summed and stored in a to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    x: str, optional
        Column label of the x-directional axis to interpolate over. Defaults to the x-column detected in `data`.
    y: str, optional
        Column label of the y-directional axis to interpolate over. Defaults to the y-column detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    integral_samples: int, optional
        The number of sample points to take when approximating the 2D column kernel.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of y_pixels which
        preserves the aspect ratio of the data.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of x_pixels which
        preserves the aspect ratio of the data.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space). Defaults to the lower bound
        of x detected in `data` snapped to the nearest integer.
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space). Defaults to the upper bound
        of x detected in `data` snapped to the nearest integer.
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space). Defaults to the lower bound
        of y detected in `data` snapped to the nearest integer.
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space). Defaults to the upper bound
        of y detected in `data` snapped to the nearest integer.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.

    Returns
    -------
    Figure
        The resulting matplotlib figure, containing the 2d render and
        a color bar indicating the magnitude of the target variable.
    Axes
        The resulting matplotlib axes, which contain the 2d rendered image.

    Raises
    -------
    ValueError
        If `x_pixels` or `y_pixels` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_3d(data, target, x, y, kernel, integral_samples, rotation, origin, x_pixels, y_pixels, x_min,
                         x_max, y_min, y_max)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    graphic = ax.imshow(image, cmap=cmap, **kwargs)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        colorbar.ax.set_ylabel(f"column {target}")

    return ax


def render_3d_cross(data: 'SarracenDataFrame',
                    target: str,
                    z_slice: float = None,
                    x: str = None,
                    y: str = None,
                    z: str = None,
                    kernel: BaseKernel = None,
                    rotation: np.ndarray = None,
                    origin: np.ndarray = None,
                    x_pixels: int = None,
                    y_pixels: int = None,
                    x_min: float = None,
                    x_max: float = None,
                    y_min: float = None,
                    y_max: float = None,
                    cmap: Union[str, Colormap] = 'RdBu',
                    cbar: bool = True,
                    cbar_kws: dict = {},
                    cbar_ax: Axes = None,
                    ax: Axes = None,
                    **kwargs) -> Axes:
    """ Render 3D particle data to a 2D grid, using a 3D cross-section.

    Render the data within a SarracenDataFrame to a 2D matplotlib object, using a 3D -> 2D
    cross-section of the target variable. The cross-section is taken of the 3D data at a specific
    value of z, and the contributions of particles near the plane are interpolated to a 2D grid.

    Parameters
    ----------
    data : SarracenDataFrame
        Particle data, in a SarracenDataFrame.
    target: str
        Column label of the target smoothing data.
    z_slice: float, optional
        Z-axis value to take the cross-section at. Defaults to the average z position in `data`.
    x: str, optional
        Column label of the x-directional axis. Defaults to the x-column detected in `data`.
    y: str, optional
        Column label of the y-directional axis. Defaults to the y-column detected in `data`.
    z: str
        Column label of the z-directional axis. Defaults to the z-column detected in `data`.
    kernel: BaseKernel, optional
        Kernel to use for smoothing the target data. Defaults to the kernel specified in `data`.
    rotation: array_like or Rotation, optional
        The rotation to apply to the data before interpolation. If defined as an array, the
        order of rotations is [z, y, x] in degrees.
    origin: array_like, optional
        Point of rotation of the data, in [x, y, z] form. Defaults to the centre
        point of the bounds of the data.
    x_pixels: int, optional
        Number of pixels in the output image in the x-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of y_pixels which
        preserves the aspect ratio of the data.
    y_pixels: int, optional
        Number of pixels in the output image in the y-direction. If both x_pixels and y_pixels are
        None, this defaults to 256. Otherwise, this value defaults to a multiple of x_pixels which
        preserves the aspect ratio of the data.
    x_min: float, optional
        Minimum bound in the x-direction (in particle data space). Defaults to the lower bound
        of x detected in `data` snapped to the nearest integer.
    x_max: float, optional
        Maximum bound in the x-direction (in particle data space). Defaults to the upper bound
        of x detected in `data` snapped to the nearest integer.
    y_min: float, optional
        Minimum bound in the y-direction (in particle data space). Defaults to the lower bound
        of y detected in `data` snapped to the nearest integer.
    y_max: float, optional
        Maximum bound in the y-direction (in particle data space). Defaults to the upper bound
        of y detected in `data` snapped to the nearest integer.
    cmap: str or Colormap, optional
        The color map to use when plotting this data.

    Returns
    -------
    Figure
        The resulting matplotlib figure, containing the 3d-cross section and
        a color bar indicating the magnitude of the target variable.
    Axes
        The resulting matplotlib axes, which contains the 3d-cross section image.

    Raises
    -------
    ValueError
        If `pixwidthx`, `pixwidthy`, `pixcountx`, or `pixcounty` are less than or equal to zero, or
        if the specified `x` and `y` minimum and maximums result in an invalid region, or
        if the provided data is not 3-dimensional.
    KeyError
        If `target`, `x`, `y`, `z`, mass, density, or smoothing length columns do not
        exist in `data`.
    """
    image = interpolate_3d_cross(data, target, z_slice, x, y, z, kernel, rotation, origin, x_pixels, y_pixels, x_min,
                               x_max, y_min, y_max)

    if ax is None:
        ax = plt.gca()

    x, y = _default_axes(data, x, y)
    x_min, x_max, y_min, y_max = _default_bounds(data, x, y, x_min, x_max, y_min, y_max)

    kwargs.setdefault("origin", 'lower')
    kwargs.setdefault("extent", [x_min, x_max, y_min, y_max])
    graphic = ax.imshow(image, cmap=cmap, **kwargs)

    # remove the x & y ticks if the data is rotated, since these no longer have physical
    # relevance to the displayed data.
    if rotation is not None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    if cbar:
        colorbar = ax.figure.colorbar(graphic, cbar_ax, ax, **cbar_kws)
        colorbar.ax.set_ylabel(target)

    return ax
