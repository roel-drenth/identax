"""Error metrics for comparing measured and simulated outputs.

All metrics are computed per output channel over the sample axis, for
arrays of shape (N, ny). Passing lists of arrays (one per experiment)
returns a list with the metric per experiment.

- ``MSE``: mean squared error.
- ``RMSE``: root mean squared error.
- ``NRMSE``: RMSE divided by the standard deviation of the measured output.
- ``BFR``: best fit rate in percent, ``100 * (1 - NRMSE)``.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array


def RMSE(y_true: Array | Sequence[Array], y_hat: Array | Sequence[Array]):
    """Calculates the Root Mean Squared Error (RMSE) per channel

    Parameters
    ----------
    y_true : Array | Sequence[Array]
        Array of true values, or a list with one array per experiment.
    y_hat : Array | Sequence[Array]
        Array of estimated values, or a list with one array per experiment.

    Returns
    -------
    RMSE: Array
        Array of RMSE, a list per experiment for list inputs
    """
    if isinstance(y_true, (list, tuple)):
        return [RMSE(yt, yh) for yt, yh in zip(y_true, y_hat)]
    return jnp.sqrt(jnp.mean((y_true - y_hat) ** 2, axis=0))


def MSE(y_true: Array | Sequence[Array], y_hat: Array | Sequence[Array]):
    """Calculates the Mean Squared Error (MSE) per channel

    Parameters
    ----------
    y_true : Array | Sequence[Array]
        Array of true values, or a list with one array per experiment.
    y_hat : Array | Sequence[Array]
        Array of estimated values, or a list with one array per experiment.

    Returns
    -------
    MSE: Array
        Array of MSE, a list per experiment for list inputs
    """
    if isinstance(y_true, (list, tuple)):
        return [MSE(yt, yh) for yt, yh in zip(y_true, y_hat)]
    return jnp.mean((y_true - y_hat) ** 2, axis=0)


def NRMSE(y_true: Array | Sequence[Array], y_hat: Array | Sequence[Array]):
    """Calculates the Normalized Root Mean Squared Error (NRMSE) per channel

    Parameters
    ----------
    y_true : Array | Sequence[Array]
        Array of true values, or a list with one array per experiment.
    y_hat : Array | Sequence[Array]
        Array of estimated values, or a list with one array per experiment.

    Returns
    -------
    NRMSE: Array
        Array of NRMSE, a list per experiment for list inputs
    """
    if isinstance(y_true, (list, tuple)):
        return [NRMSE(yt, yh) for yt, yh in zip(y_true, y_hat)]
    rmse = RMSE(y_true, y_hat)
    std = jnp.std(y_true, axis=0)
    if jnp.any(std == 0):
        raise ValueError(
            "Cannot compute NRMSE: std(y_true, axis=0) contains zeros, check input data"
        )
    return rmse / std


def BFR(y_true: Array | Sequence[Array], y_hat: Array | Sequence[Array]):
    """Calculates the Best Fit Rate (BFR) per channel

    Parameters
    ----------
    y_true : Array | Sequence[Array]
        Array of true values, or a list with one array per experiment.
    y_hat : Array | Sequence[Array]
        Array of estimated values, or a list with one array per experiment.

    Returns
    -------
    BFR: Array
        Array of BFR, a list per experiment for list inputs
    """
    if isinstance(y_true, (list, tuple)):
        return [BFR(yt, yh) for yt, yh in zip(y_true, y_hat)]
    nrmse = NRMSE(y_true, y_hat)
    return 100 * (1 - nrmse)
