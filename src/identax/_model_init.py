"""Initialization and rescaling of model parameters (internal).

These helpers create the initial LTI and LFR matrices of the models, and
rescale the latent signal z of NL-LFR models to unit standard deviation.
They are used by the model constructors and ``ModelNLLFR.rescale``.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray


def init_lti_params(
    nx: int, nu: int, ny: int, key: PRNGKeyArray
) -> tuple[Array, Array, Array, Array]:
    """Initializes parameters for LTI state-space model.

    A is set to ``0.5 * I`` (stable, with all poles at 0.5), B_u and C_y
    are drawn from a standard normal distribution and D_yu is zero.

    Parameters
    ----------
    nx  : int
        Number of states.
    nu  : int
        Number of inputs.
    ny  : int
        Number of outputs
    key : PRNGKeyArray
        Key used for random initialization.

    Returns
    -------
    A   : Array, shape (nx, nx)

    B_u : Array, shape (nx, nu)

    C_y : Array, shape (ny, nx)

    D_yu : Array, shape (ny, nu)
    """

    Bkey, Ckey = jax.random.split(key, 2)

    A = 0.5 * jnp.eye(nx, nx)
    B_u = jax.random.normal(Bkey, (nx, nu))
    C_y = jax.random.normal(Ckey, (ny, nx))
    D_yu = jnp.zeros((ny, nu))
    return A, B_u, C_y, D_yu


def init_lfr_params(
    nx: int, nu: int, ny: int, nw: int, key: PRNGKeyArray, nz: int | None = None
) -> tuple[Array, ...]:
    """Initializes parameters for model with linear fractional representation (Except for D_zw).

    The LTI part is initialized by ``init_lti_params``. B_w and D_yw are
    zero, so the initial model is the LTI model; C_z and D_zu are drawn
    uniformly from [-0.1, 0.1].

    Parameters
    ----------
    nx : int
        Number of states.
    nu : int
        Number of inputs.
    ny : int
        Number of outputs.
    nw : int
        Number of latent variables w (inputs to the LTI part).
    key : PRNGKeyArray
        Key used for random initialization.
    nz : int | None
        Number of latent variables z (outputs of the LTI part). Defaults to
        ``nw``, as required by the LPV-LFR models.

    Returns
    -------
    A   : Array, shape (nx, nx)

    B_w : Array, shape (nx, nw)

    B_u : Array, shape (nx, nu)

    C_z : Array, shape (nz, nx)

    D_zu : Array, shape (nz, nu)

    C_y : Array, shape (ny, nx)

    D_yw : Array, shape (ny, nw)

    D_yu : Array, shape (ny, nu)
    """
    if nz is None:
        nz = nw

    LTIkey, Czkey, Dzukey = jax.random.split(key, 3)

    A, B_u, C_y, D_yu = init_lti_params(nx, nu, ny, LTIkey)
    B_w = jnp.zeros((nx, nw))
    C_z = jax.random.uniform(Czkey, (nz, nx), minval=-0.1, maxval=0.1)
    D_zu = jax.random.uniform(Dzukey, (nz, nu), minval=-0.1, maxval=0.1)
    D_yw = jnp.zeros((ny, nw))
    return A, B_w, B_u, C_z, D_zu, C_y, D_yw, D_yu


def scale_lti_ss(
    A: Array, B: Array, C: Array, D: Array, x: Array
) -> tuple[Array, Array, Array, Array]:
    """Change state coordinates so that each state has unit standard deviation.

    Applies the similarity transform ``x_tilde = T x`` with
    ``T = diag(std(x))^-1``, which leaves the input-output behavior of the
    model unchanged. Typically used on a trained LTI model before passing
    its matrices as ``init_sys`` to an LFR model, so that the nonlinear part
    starts from well-scaled states::

        A, B_u, C_y, D_yu = idx.scale_lti_ss(
            lti.A, lti.B_u, lti.C_y, lti.D_yu, x_sim
        )
        model = idx.ModelAffineLPVLFR(config, key, init_sys=(A, B_u, C_y, D_yu))

    Parameters
    ----------
    A : Array, shape (nx, nx)
    B : Array, shape (nx, nu)
    C : Array, shape (ny, nx)
    D : Array, shape (ny, nu)
    x : Array, shape (N, nx)
        Simulated state trajectory of the model, e.g. the second output of
        ``simulate``.

    Returns
    -------
    A_tilde : Array, shape (nx, nx)
    B_tilde : Array, shape (nx, nu)
    C_tilde : Array, shape (ny, nx)
    D : Array, shape (ny, nu)
        Unchanged.

    Raises
    ------
    ValueError
        If a state has zero standard deviation.
    """
    stds = jnp.std(x, 0)
    if jnp.any(stds == 0):
        raise ValueError(
            "Cannot scale state-space: std(x, 0) contains zeros, check state data"
        )
    Tinv = jnp.diag(stds)
    T = jnp.linalg.inv(Tinv)
    Atilde = T @ A @ Tinv
    Btilde = T @ B
    Ctilde = C @ Tinv
    return Atilde, Btilde, Ctilde, D


def scale_nllfr(C_z: Array, D_zu: Array, z: Array) -> tuple[Array, Array]:
    """Rescale the z equation so that each latent signal z has unit standard deviation.

    The rows of C_z and D_zu are divided by the standard deviation of the
    corresponding channel of z. This is a change of coordinates of z, so
    the model's input-output behavior is unchanged up to the nonlinearity's
    input scaling.

    Parameters
    ----------
    C_z : Array, shape (nz, nx)
    D_zu : Array, shape (nz, nu)
    z : Array, shape (N, nz)
        Simulated latent signal used to compute the standard deviations.

    Returns
    -------
    C_z_scaled : Array, shape (nz, nx)
    D_zu_scaled : Array, shape (nz, nu)

    Raises
    ------
    ValueError
        If a channel of z has zero standard deviation. Because of this
        check, the function cannot be traced (jit or vmap).
    """
    stds = jnp.std(z, 0)
    if jnp.any(stds == 0):
        raise ValueError(
            "Cannot scale model: std(z, 0) contains zeros, check latent variable data"
        )
    Tinv = jnp.diag(stds)
    T = jnp.linalg.inv(Tinv)
    C_z_scaled = T @ C_z
    D_zu_scaled = T @ D_zu
    return C_z_scaled, D_zu_scaled
