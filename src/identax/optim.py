"""Training of models and ensembles by prediction-error minimization.

The main entry points are:

- ``optimize``: train a single model.
- ``optimize_ensemble``: train an ensemble of models in parallel (vmapped).
- ``learn_initial_state``: estimate the initial state of a trained model for
  new data.

Training is a two-step procedure: a fixed number of Adam iterations gives a
warm start for L-BFGS, which converges to the final minimum. Both steps are
configured by ``OptimConfig``. The cost is the mean squared simulation error
on scaled data, using single shooting, unconstrained multiple shooting (UMS)
or encoder-based multiple shooting (EBMS) (see ``ShootingType``), plus
optional l2-regularization. The full loss history of both steps is returned
in an ``OptimHistory``.

The remaining functions are building blocks used by these entry points.
"""

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from itertools import accumulate
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from identax.dataprocessing import (
    as_collection,
    concatenate_iddata,
    create_encoder_iddata,
    create_multiple_shooting_iddata,
    create_padded_iddata,
    iddata,
    iddata_collection,
    initialize_scalers,
    truncate_iddata,
)


# Line-search step limit used during the LBFGS warmup phase (see OptimConfig).
_LBFGS_WARMUP_MAX_LS = 30


class ShootingType(StrEnum):
    """How the simulation error is computed during training.

    - ``SINGLE``: simulate each experiment from its start. The initial state
      of each experiment is optimized together with the model parameters.
    - ``MULTIPLE``: unconstrained multiple shooting (UMS). Cut the data into
      sub-records of length ``T`` and simulate each sub-record from its own
      optimized initial state.
    - ``ENCODER``: encoder-based multiple shooting (EBMS). Cut the data into
      sub-records of length ``T`` and estimate each sub-record's initial
      state with the model's subspace encoder, from the ``n_hist`` samples
      before it.
    """

    SINGLE = "single"
    MULTIPLE = "multiple"
    ENCODER = "encoder"


@dataclass(frozen=True, kw_only=True)
class OptimConfig:
    """Configuration for the optimization-based identification procedure

    Parameters
    ----------
    adam_epochs          : int
            Number of Adam optimization epochs. Set to 0 to skip Adam.
    lbfgs_epochs         : int
            Number of LBFGS optimization epochs. Set to 0 to skip LBFGS.
    l2_rho               : float
            L2 regularization strength applied to the trainable parameters.
    adam_lr              : float
            Adam learning rate.
    shooting_type        : ShootingType | str
            Type of shooting used during training: "single", "multiple"
            (UMS) or "encoder" (EBMS, requires a model with a state encoder).
    T                    : int | None
            Length of each sub-record. Required for multiple and encoder
            shooting, not used for single shooting.
    overlap              : int
            Overlap between consecutive sub-records, must be smaller than T.
    freeze_lti           : bool
            When True, the LTI parameters (A, B, C, D) are frozen during
            optimization.
    lbfgs_memory         : int
            Number of past iterates retained by the LBFGS Hessian approximation.
    lbfgs_max_ls         : int
            Maximum number of line-search steps per LBFGS iteration. When
            ``lbfgs_warmup`` is True, this is the limit used after the warmup
            phase; set it lower than the warmup limit (30) to reduce wasted
            computation in vmapped ensemble runs.
    lbfgs_warmup         : bool
            When True (default), the first ``lbfgs_memory`` LBFGS iterations
            use a line-search limit of 30 steps, to build a good
            inverse-Hessian approximation, then switch to ``lbfgs_max_ls``
            for the remaining iterations. The LBFGS Hessian history is
            preserved across the switch. Set to False to use
            ``lbfgs_max_ls`` for all iterations.
    grad_tol             : float
            Gradient-norm tolerance below which LBFGS training stops. Frozen
            parameters do not count towards the gradient norm.
    lbfgs_ls_tol         : float
            Convergence tolerance of the zoom line search.
    lbfgs_slope_rtol     : float
            Relative tolerance on the slope (Wolfe) condition of the line search.
    lbfgs_approx_dec_rtol: float | None
            Relative tolerance on the approximate decrease of the line search.
            When None, the check is disabled.
    lbfgs_loss_max       : float
            LBFGS training aborts when the loss exceeds this value.
    lbfgs_stagnation_tol : float
            LBFGS training aborts when the absolute loss change between
            successive iterations falls below this value.
    """

    adam_epochs: int = 1000
    lbfgs_epochs: int = 2000
    l2_rho: float = 0.0
    adam_lr: float = 1e-3
    shooting_type: ShootingType | str = ShootingType.SINGLE
    T: int | None = None
    overlap: int = 0
    freeze_lti: bool = False
    lbfgs_memory: int = 20
    lbfgs_max_ls: int = 20
    lbfgs_warmup: bool = True
    grad_tol: float = 1e-5
    lbfgs_ls_tol: float = 1e-8
    lbfgs_slope_rtol: float = 1e-4
    lbfgs_approx_dec_rtol: float | None = None
    lbfgs_loss_max: float = 1e6
    lbfgs_stagnation_tol: float = 2.2e-16

    def __post_init__(self) -> None:
        if self.adam_epochs < 0:
            raise ValueError(f"adam_epochs must be >= 0, got {self.adam_epochs} ")
        if self.lbfgs_epochs < 0:
            raise ValueError(f"lbfgs_epochs must be >= 0, got {self.lbfgs_epochs} ")
        if self.l2_rho < 0:
            raise ValueError(f"l2_rho must be >= 0, got {self.l2_rho}")
        if self.adam_lr <= 0:
            raise ValueError(f"adam_lr must be > 0, got {self.adam_lr}")
        if self.lbfgs_memory <= 0 and self.lbfgs_epochs > 0:
            raise ValueError(
                f"When LBFGS is used, lbfgs_memory must be > 0, got {self.lbfgs_memory}"
            )
        if self.lbfgs_max_ls <= 0 and self.lbfgs_epochs > 0:
            raise ValueError(
                f"When LBFGS is used, lbfgs_max_ls must be > 0, got {self.lbfgs_max_ls}"
            )
        if self.lbfgs_ls_tol < 0:
            raise ValueError(f"lbfgs_ls_tol must be >= 0, got {self.lbfgs_ls_tol}")
        if self.lbfgs_slope_rtol <= 0:
            raise ValueError(
                f"lbfgs_slope_rtol must be > 0, got {self.lbfgs_slope_rtol}"
            )
        if self.lbfgs_approx_dec_rtol is not None and self.lbfgs_approx_dec_rtol < 0:
            raise ValueError(
                f"lbfgs_approx_dec_rtol must be None or >= 0, "
                f"got {self.lbfgs_approx_dec_rtol}"
            )
        if self.lbfgs_loss_max <= 0:
            raise ValueError(f"lbfgs_loss_max must be > 0, got {self.lbfgs_loss_max}")
        if self.lbfgs_stagnation_tol < 0:
            raise ValueError(
                f"lbfgs_stagnation_tol must be >= 0, got {self.lbfgs_stagnation_tol}"
            )
        if self.grad_tol <= 0:
            raise ValueError(f"grad_tol must be > 0, got {self.grad_tol}")
        # Validate and normalize the shooting type, so that typos raise here
        object.__setattr__(self, "shooting_type", ShootingType(self.shooting_type))
        if self.T is not None and self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.shooting_type != ShootingType.SINGLE:
            if self.T is None:
                raise ValueError(
                    f"shooting_type='{self.shooting_type}' requires the "
                    "sub-record length T to be set"
                )
            if self.overlap >= self.T:
                raise ValueError(
                    f"overlap must be < T, got overlap={self.overlap} and T={self.T}"
                )


@dataclass(frozen=True)
class OptimHistory:
    """Recorded optimization results returned by optimize() / optimize_ensemble().

    Parameters
    ----------
    train            : Array
            Combined Adam + LBFGS training loss, concatenated along axis 0.
    val              : Array
            Combined Adam + LBFGS validation loss, concatenated along axis 0.
            Empty when no validation data was provided.
    adam_train       : Array
            Per-iteration training loss during the Adam phase. Empty if Adam
            was skipped.
    lbfgs_train      : Array
            Per-iteration training loss during the LBFGS phase. Empty if LBFGS
            was skipped.
    adam_val         : Array
            Per-iteration validation loss during the Adam phase.
    lbfgs_val        : Array
            Per-iteration validation loss during the LBFGS phase.
    adam_time        : float
            Wall-clock time (seconds) spent in the Adam phase. 0.0 if skipped.
    lbfgs_time       : float
            Wall-clock time (seconds) spent in the LBFGS phase. 0.0 if skipped.
    config           : OptimConfig
            Configuration used for this optimization run.
    timestamp        : str
            ISO-format timestamp captured at the start of the optimization run.
    lbfgs_grad_norm  : Array
            Per-iteration infinity-norm of the gradient of the trainable
            variables during the LBFGS phase. Empty if LBFGS was skipped.
    lbfgs_n_iters    : int | Array
            Number of LBFGS iterations actually completed (may be less than
            lbfgs_epochs due to early stopping). 0 if LBFGS was skipped. For
            ensembles, an array of shape (E,).
    lbfgs_stop_reason: int | Array
            Reason the LBFGS loop terminated. Encoded as an int:
            0 = not run, 1 = converged (grad_norm <= grad_tol),
            2 = max iterations reached, 3 = stagnation,
            4 = diverged (NaN/inf loss), 5 = loss exceeded loss_max.
            For ensembles, an array of shape (E,).
    total_time       : float
            Total wall-clock time (seconds) across all phases. Computed
            automatically from adam_time + lbfgs_time.
    """

    train: Array
    val: Array
    adam_train: Array
    lbfgs_train: Array
    adam_val: Array
    lbfgs_val: Array
    adam_time: float
    lbfgs_time: float
    config: OptimConfig
    timestamp: str
    lbfgs_grad_norm: Array = field(default_factory=lambda: jnp.empty((0,)))
    lbfgs_n_iters: int | Array = 0
    lbfgs_stop_reason: int | Array = 0
    total_time: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_time", self.adam_time + self.lbfgs_time)


def l2_regularization(params, frozen: PyTree | None = None):
    """Sum of squared parameters, leaving out leaves marked True in frozen."""
    params = eqx.filter(params, eqx.is_array)
    if frozen is not None:
        params = jax.tree.map(lambda x, f: None if f else x, params, frozen)
    return jax.tree.reduce(lambda acc, x: acc + jnp.sum(x**2), params, 0.0)


def _l2_penalty(params, freeze_mask: PyTree | None, l2_rho):
    """L2 penalty on the trainable parameters, freeze_mask covers (params, x0)."""
    frozen = None if freeze_mask is None else freeze_mask[0]
    return l2_rho * l2_regularization(params, frozen)


def _trainable_grad_norm(grads, freeze_mask: PyTree | None):
    """Infinity-norm of the gradient, ignoring parameters that are not updated."""
    if freeze_mask is not None:
        grads = jax.tree.map(
            lambda g, f: jnp.zeros_like(g) if f else g, grads, freeze_mask
        )
    return optax.tree_utils.tree_norm(grads, ord="inf")


def mse(Y: Array, Yhat: Array, mask: Array | None = None):
    """Mean squared error over all samples, only counting samples where mask is True."""
    if mask is None:
        return jnp.mean((Y - Yhat) ** 2)
    # Mask the error before squaring, so padded samples get zero gradient
    err = jnp.where(mask[..., None], Y - Yhat, 0.0)
    return jnp.sum(err**2) / (jnp.sum(mask) * Y.shape[-1])


def val_single_loss_fn(opt_vars, static, val_data):
    """Validation loss: MSE of a single-shooting simulation from a zero initial state.

    Parameters
    ----------
    opt_vars : tuple
        ``(params, x0)``, the trainable part of the model and the training
        initial state(s). Only the state dimension of ``x0`` is used.
    static : Any
        Non-trainable part of the model, recombined with ``params`` using
        ``eqx.combine``.
    val_data : iddata
        Scaled validation data, of shape (N, n) or padded to (n_exp, N, n).

    Returns
    -------
    Array
        Scalar validation loss, without regularization.
    """
    params, x0 = opt_vars
    sys = eqx.combine(params, static)

    def simulate(val_data):
        Yhat, _ = sys._simulate(val_data, jnp.zeros((1, x0.shape[-1])))
        return Yhat

    if val_data.U.ndim == 2:
        Yhat = simulate(val_data)
    else:
        Yhat = jax.vmap(simulate)(val_data)

    return mse(val_data.Y, Yhat, val_data.mask)


def multiple_shooting_loss_fn(opt_vars, static, train_data, l2_rho, freeze_mask=None):
    """Training loss when the initial states are optimization variables.

    Used for both single and multiple shooting: each trajectory or window in
    ``train_data`` is simulated from its own initial state in ``x0``.

    Parameters
    ----------
    opt_vars : tuple
        ``(params, x0)``, the trainable part of the model and the initial
        states, of shape (1, nx) for a single trajectory or (R, nx) for R
        windows or padded experiments.
    static : Any
        Non-trainable part of the model.
    train_data : iddata
        Scaled training data, of shape (N, n) or (R, T, n).
    l2_rho : float
        L2 regularization strength.
    freeze_mask : PyTree | None
        Mask over ``opt_vars``; frozen parameters are left out of the L2 term.

    Returns
    -------
    Array
        Scalar loss, MSE plus L2 penalty.
    """
    params, x0 = opt_vars
    sys = eqx.combine(params, static)

    def simulate(train_data, x0):
        Yhat, _ = sys._simulate(train_data, x0)
        return Yhat

    if train_data.U.ndim == 2:
        Yhat = simulate(train_data, x0)
    else:
        Yhat = jax.vmap(simulate)(train_data, x0)

    return mse(train_data.Y, Yhat, train_data.mask) + _l2_penalty(
        params, freeze_mask, l2_rho
    )


def encoder_loss_fn(opt_vars, static, train_data, l2_rho, freeze_mask=None):
    """Training loss when initial states are predicted by the state encoder.

    The ``x0`` entry of ``opt_vars`` is ignored; each window's initial state
    is computed from its history (``U_hist``, ``Y_hist``, ``D_hist``).
    Arguments and return value are as in ``multiple_shooting_loss_fn``.
    """
    params, _ = opt_vars  # x0 generated by StateEncoder
    sys = eqx.combine(params, static)

    def simulate(td):
        x0 = sys._encoder_initial_state_prediction(td)
        Yhat, _ = sys._simulate(td, x0)
        return Yhat

    if train_data.U.ndim == 2:
        Yhat = simulate(train_data)
    else:
        Yhat = jax.vmap(simulate)(train_data)

    return mse(train_data.Y, Yhat, train_data.mask) + _l2_penalty(
        params, freeze_mask, l2_rho
    )


def encoder_loss_fn_ensemble(opt_vars, static, train_data, l2_rho, freeze_mask=None):
    """Ensemble version of ``encoder_loss_fn``.

    The encoder is reached through the ensemble's ``models`` attribute.
    Called per ensemble member under ``vmap``.
    """
    params, _ = opt_vars  # x0 generated by StateEncoder
    sys = eqx.combine(params, static)

    def simulate(td):
        x0 = sys.models._encoder_initial_state_prediction(td)
        Yhat, _ = sys._simulate(td, x0)
        return Yhat

    if train_data.U.ndim == 2:
        Yhat = simulate(train_data)
    else:
        Yhat = jax.vmap(simulate)(train_data)

    return mse(train_data.Y, Yhat, train_data.mask) + _l2_penalty(
        params, freeze_mask, l2_rho
    )


def _create_training_iddata(
    data_scaled: iddata | iddata_collection,
    config: OptimConfig,
    n_hist: int | None,
) -> tuple[iddata, list[int] | None]:
    """Pads or splits scaled training data according to the shooting type.

    Returns the training data and the number of records per experiment, which
    is None for single shooting on a single iddata (no record axis).
    """
    if config.shooting_type == ShootingType.ENCODER and n_hist is None:
        raise ValueError(
            "shooting_type='encoder' (EBMS) requires a model with a state encoder; "
            "set StateEncoder=idx.StateEncoderConfig(...) in the model config, or "
            "use shooting_type='multiple' (UMS)."
        )
    if config.shooting_type == ShootingType.SINGLE:
        if isinstance(data_scaled, iddata):
            return data_scaled, None
        return create_padded_iddata(data_scaled), [1] * len(data_scaled)

    if isinstance(data_scaled, iddata):
        data_scaled = iddata_collection([data_scaled])

    if config.shooting_type == ShootingType.MULTIPLE:
        records = [
            create_multiple_shooting_iddata(d, T=config.T, overlap=config.overlap)
            for d in data_scaled
        ]
    else:
        records = [
            create_encoder_iddata(d, T=config.T, overlap=config.overlap, n_hist=n_hist)
            for d in data_scaled
        ]

    counts = [int(n) for _, n in records]
    if min(counts) <= 0:
        raise ValueError(
            f"Experiment {counts.index(min(counts))} is too short for "
            f"{config.shooting_type} shooting with T={config.T}."
        )
    return concatenate_iddata([d for d, _ in records]), counts


def _stack_x0(x0: Array | Sequence[Array], shooting_type: ShootingType) -> Array:
    """Stacks a list of per-experiment initial states along the record axis (-2).

    For single shooting, each entry may be (nx,) or (1, nx), the shape used
    for a single iddata; both give one row per experiment.
    """
    if not isinstance(x0, (list, tuple)):
        return x0
    if shooting_type == ShootingType.SINGLE:
        x0 = [_squeeze_single_state(jnp.asarray(x))[..., None, :] for x in x0]
    return jnp.concatenate(x0, axis=-2)


def _squeeze_single_state(x: Array) -> Array:
    """Turn a single state of shape (1, nx) into (nx,); other shapes unchanged."""
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    return x


def _broadcast_x0(x0: Array, n_rec: int) -> Array:
    """Return per-record initial states of shape (n_rec, nx).

    Accepts states that are already (n_rec, nx), or a single shared state of
    shape (nx,) or (1, nx), which is repeated for every record.
    """
    if x0.ndim == 2 and x0.shape[0] == n_rec:
        return x0
    x0 = _squeeze_single_state(x0)
    if x0.ndim != 1:
        raise ValueError(
            f"x0 has shape {x0.shape}; expected a shared state of shape (nx,) "
            f"or (1, nx), or one state per record of shape ({n_rec}, nx)."
        )
    return jnp.repeat(x0[None, :], n_rec, axis=0)


def _split_x0(x0: Array, counts: list[int], shooting_type: ShootingType) -> list[Array]:
    """Splits stacked initial states (..., R, nx) into a list per experiment."""
    x0s = jnp.split(x0, list(accumulate(counts))[:-1], axis=-2)
    if shooting_type == ShootingType.SINGLE:
        x0s = [x[..., 0, :] for x in x0s]
    return x0s


def adam_training(
    opt_vars: tuple,
    static: Any,
    train_data: iddata,
    adam_epochs: int,
    adam_lr: float,
    loss_fn: Callable,
    l2_rho: float,
    val_data: iddata | None = None,
    freeze_mask: PyTree | None = None,
):
    """Run Adam for a fixed number of epochs, compiled into a single ``lax.scan``.

    Parameters
    ----------
    opt_vars : tuple
        ``(params, x0)`` to optimize.
    static : Any
        Non-trainable part of the model.
    train_data : iddata
        Scaled (and, if applicable, windowed) training data.
    adam_epochs : int
        Number of Adam steps.
    adam_lr : float
        Learning rate.
    loss_fn : Callable
        Loss with signature ``loss_fn(opt_vars, static, train_data, l2_rho,
        freeze_mask)``.
    l2_rho : float
        L2 regularization strength.
    val_data : iddata | None
        Scaled validation data. When given, the validation loss is recorded
        after every step.
    freeze_mask : PyTree | None
        Mask over ``opt_vars``; entries marked True are not updated.

    Returns
    -------
    opt_vars : tuple
        Optimized variables after the last epoch.
    losses : tuple[Array, Array]
        Training loss per epoch, and validation loss per epoch (empty without
        ``val_data``).
    """
    adam = optax.adam(adam_lr)
    adam = optax.selective_transform(adam, freeze_mask=freeze_mask)

    adam_state = adam.init(opt_vars)

    def step_fn(carry, _):
        opt_vars, adam_state = carry

        train_loss, grads = eqx.filter_value_and_grad(loss_fn)(
            opt_vars, static, train_data, l2_rho, freeze_mask
        )

        updates, adam_state = adam.update(grads, adam_state, opt_vars)
        opt_vars = optax.apply_updates(opt_vars, updates)

        if val_data is not None:
            val_loss = val_single_loss_fn(opt_vars, static, val_data)
        else:
            val_loss = jnp.empty((0,))

        return (opt_vars, adam_state), (train_loss, val_loss)

    # Run all epochs inside a single compiled scan (no per-epoch host sync)
    @eqx.filter_jit
    def run(opt_vars, adam_state):
        return jax.lax.scan(
            step_fn, init=(opt_vars, adam_state), xs=None, length=adam_epochs
        )

    (opt_vars, adam_state), (train_losses, val_losses) = run(opt_vars, adam_state)

    if val_data is None:
        val_losses = jnp.empty((0,))

    return opt_vars, (train_losses, val_losses)


@eqx.filter_jit
def lbfgs_training(
    opt_vars,
    static,
    train_data,
    lbfgs_epochs,
    loss_fn,
    l2_rho,
    memory,
    max_ls,
    warmup,
    val_data=None,
    grad_tol=1e-5,
    freeze_mask: PyTree | None = None,
    ls_tol: float = 1e-8,
    slope_rtol: float = 1e-4,
    approx_dec_rtol: float | None = None,
    loss_max: float = 1e6,
    stagnation_tol: float = 2.2e-16,
):
    """Run LBFGS with a zoom line search until a stopping criterion is met.

    The loop stops at the first of: ``lbfgs_epochs`` iterations, gradient
    infinity-norm (over the trainable variables) at most ``grad_tol``, a
    non-finite loss, a loss of at least ``loss_max``, or an absolute loss
    change of at most ``stagnation_tol``.

    With ``warmup``, the first ``min(memory, lbfgs_epochs)`` iterations use a
    line-search limit of ``_LBFGS_WARMUP_MAX_LS`` (30) steps, after which
    ``max_ls`` is used. The LBFGS
    state is kept across the switch.

    When ``val_data`` is given, the parameters with the lowest validation
    loss are returned instead of the final ones.

    Parameters
    ----------
    opt_vars, static, train_data, loss_fn, l2_rho, val_data, freeze_mask
        As in ``adam_training``.
    lbfgs_epochs : int
        Maximum number of LBFGS iterations.
    memory : int
        Number of past updates kept in the inverse-Hessian approximation.
    max_ls : int
        Maximum number of line-search steps per iteration (after warmup).
    warmup : bool
        Whether to use the warmup phase described above.
    grad_tol, ls_tol, slope_rtol, approx_dec_rtol, loss_max, stagnation_tol
        Tolerances; see the corresponding ``OptimConfig`` fields.

    Returns
    -------
    opt_vars : tuple
        Optimized variables.
    results : tuple
        ``(train_losses, val_losses, grad_norms, n_iters, stop_reason)``.
        The per-iteration arrays have length ``lbfgs_epochs`` and are NaN
        after the last completed iteration. ``stop_reason`` uses the codes
        of ``OptimHistory.lbfgs_stop_reason``.
    """

    # Manually recreate lbfgs, wrapping optax.selective_transform  does not work with optax.lbfgs
    def make_lbfgs(mls):
        return optax.chain(
            optax.selective_transform(
                optax.scale_by_lbfgs(memory_size=memory), freeze_mask=freeze_mask
            ),
            optax.scale(-1.0),
            optax.scale_by_zoom_linesearch(
                initial_guess_strategy="one",
                max_linesearch_steps=mls,
                verbose=False,
                tol=ls_tol,
                approx_dec_rtol=approx_dec_rtol,
                slope_rtol=slope_rtol,
            ),
        )

    # Two optimizers that differ only in max_linesearch_steps share the same
    # state structure, so a single opt_state can flow through both phases.
    lbfgs_main = make_lbfgs(max_ls)
    opt_state = lbfgs_main.init(opt_vars)

    if warmup:
        lbfgs_warmup = make_lbfgs(_LBFGS_WARMUP_MAX_LS)
        warmup_iters = min(memory, lbfgs_epochs)
    else:
        warmup_iters = 0

    def value_fn(params):
        return loss_fn(params, static, train_data, l2_rho, freeze_mask)

    value_and_grad = optax.value_and_grad_from_state(value_fn)

    train_losses = jnp.full((lbfgs_epochs,), jnp.nan)
    grad_norms = jnp.full((lbfgs_epochs,), jnp.nan)

    if val_data is None:
        val_losses = jnp.empty((0,))
    else:
        val_losses = jnp.full((lbfgs_epochs,), jnp.nan)

    best_val_loss = jnp.inf
    best_params = opt_vars

    def make_cond_fn(max_epochs):
        def cond_fn(carry):
            (
                i,
                params,
                opt_state,
                train_losses,
                val_losses,
                grad_norms,
                best_val_loss,
                best_params,
                grad_norm,
            ) = carry

            return (
                (i < max_epochs)
                & (grad_norm > grad_tol)
                & ((i == 0) | jnp.isfinite(train_losses[i - 1]))
                & ((i == 0) | (train_losses[i - 1] < loss_max))
                & (
                    (i <= 1)
                    | (
                        jnp.abs(train_losses[i - 1] - train_losses[i - 2])
                        > stagnation_tol
                    )
                )
            )

        return cond_fn

    def make_body_fn(lbfgs_opt):
        def body_fn(carry):
            (
                i,
                params,
                opt_state,
                train_losses,
                val_losses,
                grad_norms,
                best_val_loss,
                best_params,
                _,
            ) = carry

            # compute gradient
            loss, grads = value_and_grad(params, state=opt_state)

            # LBFGS step
            updates, opt_state = lbfgs_opt.update(
                grads,
                opt_state,
                params=params,
                value=loss,
                grad=grads,
                value_fn=value_fn,
            )

            params = optax.apply_updates(params, updates)

            # evaluate new point
            loss, grads = value_and_grad(params, state=opt_state)
            grad_norm = _trainable_grad_norm(grads, freeze_mask)

            train_losses = train_losses.at[i].set(loss)
            grad_norms = grad_norms.at[i].set(grad_norm)

            # validation
            if val_data is not None:
                val_loss = val_single_loss_fn(params, static, val_data)

                val_losses = val_losses.at[i].set(val_loss)

                best_val_loss, best_params = jax.lax.cond(
                    val_loss < best_val_loss,
                    lambda _: (val_loss, params),
                    lambda _: (best_val_loss, best_params),
                    operand=None,
                )

            return (
                i + 1,
                params,
                opt_state,
                train_losses,
                val_losses,
                grad_norms,
                best_val_loss,
                best_params,
                grad_norm,
            )

        return body_fn

    init_state = (
        0,
        opt_vars,
        opt_state,
        train_losses,
        val_losses,
        grad_norms,
        best_val_loss,
        best_params,
        jnp.inf,
    )

    # Warmup phase: high max_ls for the first warmup_iters iterations to
    # build a good inverse-Hessian approximation.
    if warmup:
        carry = jax.lax.while_loop(
            make_cond_fn(warmup_iters),
            make_body_fn(lbfgs_warmup),
            init_state,
        )
    else:
        carry = init_state

    final_state = jax.lax.while_loop(
        make_cond_fn(lbfgs_epochs),
        make_body_fn(lbfgs_main),
        carry,
    )

    (
        n_iters,
        params,
        opt_state,
        train_losses,
        val_losses,
        grad_norms,
        best_val_loss,
        best_params,
        grad_norm,
    ) = final_state

    if val_data is not None:
        params = best_params

    # Determine why the LBFGS loop stopped, checking conditions in the same
    # order as cond_fn so the first failing condition is reported.
    # 0 = not run, 1 = converged, 2 = max iterations, 3 = stagnation,
    # 4 = diverged, 5 = loss_max
    last_loss = jnp.where(n_iters > 0, train_losses[n_iters - 1], jnp.nan)
    stop_reason = jnp.where(
        n_iters >= lbfgs_epochs,
        2,
        jnp.where(
            grad_norm <= grad_tol,
            1,
            jnp.where(
                ~jnp.isfinite(last_loss),
                4,
                jnp.where(
                    last_loss >= loss_max,
                    5,
                    3,
                ),
            ),
        ),
    )

    return params, (train_losses, val_losses, grad_norms, n_iters, stop_reason)


def adam_training_ensemble(
    opt_vars: tuple,
    static: Any,
    train_data: eqx.Module,
    adam_epochs: int,
    adam_lr: float,
    loss_fn: Callable,
    l2_rho: float,
    val_data: iddata | None = None,
    freeze_mask: PyTree | None = None,
):
    """Ensemble version of ``adam_training``.

    ``opt_vars`` carries a leading ensemble axis of size E. Losses and
    gradients are computed per member with ``vmap`` on shared training and
    validation data, and the members are updated together.

    Returns
    -------
    opt_vars : tuple
        Optimized variables, with the leading ensemble axis.
    losses : tuple[Array, Array]
        Training and validation loss, of shape (adam_epochs, E) (validation
        empty without ``val_data``).
    """
    adam = optax.adam(adam_lr)
    adam = optax.selective_transform(adam, freeze_mask=freeze_mask)

    adam_state = adam.init(opt_vars)

    @eqx.filter_jit
    def step_fn(carry, _):
        opt_vars, adam_state = carry

        train_loss, grads = eqx.filter_vmap(
            eqx.filter_value_and_grad(loss_fn), in_axes=(0, None, None, None, None)
        )(opt_vars, static, train_data, l2_rho, freeze_mask)

        updates, adam_state = adam.update(grads, adam_state, opt_vars)
        opt_vars = optax.apply_updates(opt_vars, updates)

        if val_data is not None:
            val_loss = eqx.filter_vmap(val_step, in_axes=(0))(opt_vars)
        else:
            val_loss = jnp.empty((0,))

        new_carry = (opt_vars, adam_state)
        return new_carry, (train_loss, val_loss)

    @eqx.filter_jit
    def val_step(opt_vars):
        return val_single_loss_fn(opt_vars, static, val_data)

    (opt_vars, adam_state), (train_loss, val_loss) = jax.lax.scan(
        step_fn, init=(opt_vars, adam_state), xs=None, length=adam_epochs
    )

    return opt_vars, (train_loss, val_loss)


def _scale_validation_iddata(
    scalers, val_data: iddata | iddata_collection | Sequence[iddata] | None
) -> iddata | None:
    """Scale validation data; several experiments are padded into one iddata."""
    if val_data is None:
        return None
    if isinstance(val_data, iddata):
        return scalers.transform(val_data)
    return create_padded_iddata(scalers.transform(as_collection(val_data)))


def optimize(
    sys: eqx.Module,
    train_data: iddata | iddata_collection | Sequence[iddata],
    x0: Array | Sequence[Array],
    config: OptimConfig,
    *,
    val_data: iddata | iddata_collection | Sequence[iddata] | None = None,
):
    """Optimizes the model parameters and initial states on the training data.

    Runs ``config.adam_epochs`` Adam steps followed by up to
    ``config.lbfgs_epochs`` LBFGS iterations. If the model's scalers are not
    fitted yet, they are first fitted on ``train_data``. Scaler parameters
    are never trained; with ``config.freeze_lti``, the LTI matrices are not
    trained either.

    Parameters
    ----------
    sys : eqx.Module
        Model to train, e.g. ``ModelLinStateSpace`` or ``ModelAffineLPVLFR``.
    train_data : iddata | iddata_collection | Sequence[iddata]
        Training data in physical units: a single experiment, or several
        experiments of possibly different lengths.
    x0 : Array | Sequence[Array]
        Initial guess for the initial state(s). Accepted forms:

        - a single state of shape (nx,) or (1, nx), shared by all
          experiments and records;
        - for multiple experiments, a list with one entry per experiment,
          of shape (nx,) or (1, nx) for single shooting and (N_rec_i, nx)
          for multiple shooting (as returned by a previous run);
        - for multiple shooting, stacked states of shape (N_rec, nx).

        Ignored for encoder shooting.
    config : OptimConfig
        Training settings.
    val_data : iddata | iddata_collection | Sequence[iddata] | None
        Validation data in physical units, simulated from a zero initial
        state. When given, the LBFGS phase returns the parameters with the
        lowest validation loss.

    Returns
    -------
    sys : eqx.Module
        Trained model, with fitted scalers.
    x0 : Array | list[Array]
        Learned initial state(s). For a single experiment, the shape of
        the given ``x0`` for single shooting, or (N_rec, nx) for multiple
        shooting.
        For multiple experiments, a list with one entry per experiment, of
        shape (nx,) for single shooting or (N_rec_i, nx) for multiple
        shooting.
    history : OptimHistory
        Losses, timings and LBFGS stopping information.
    """

    timestamp = datetime.now().isoformat()

    is_collection = not isinstance(train_data, iddata)
    if is_collection:
        train_data = as_collection(train_data)

    # Initialize scalers using training data, if necessary
    if not sys.scalers.is_initialized:
        warnings.warn(
            "DataScaler not yet initialized, initializing using training data.",
            stacklevel=2,
        )
        sys = initialize_scalers(sys, train_data)

    train_data_scaled = sys.scalers.transform(train_data)
    val_data_scaled = _scale_validation_iddata(sys.scalers, val_data)

    train_data_scaled, counts = _create_training_iddata(
        train_data_scaled, config, getattr(sys, "n_hist", None)
    )

    loss_fn = multiple_shooting_loss_fn
    if config.shooting_type == ShootingType.ENCODER:
        loss_fn = encoder_loss_fn
    elif counts is not None:
        # One initial state per record. Accept previously learned states of
        # shape (N_rec, nx) or a list per experiment (e.g. returned by a prior
        # optimize() run); otherwise broadcast a single state (nx,) or (1, nx).
        x0 = _broadcast_x0(_stack_x0(x0, config.shooting_type), sum(counts))

    # Filter trainable parameters
    filter_spec = jax.tree_util.tree_map(eqx.is_array, sys)  # filter out all non-arrays

    # Split model into trainable and static
    params, static = eqx.partition(sys, filter_spec)

    opt_vars = (params, x0)

    # Create mask for freezing parameters
    freeze_mask = jax.tree_util.tree_map(lambda _: False, opt_vars)
    freeze_mask = eqx.tree_at(
        lambda s: s[0].scalers,
        freeze_mask,
        jax.tree.map(lambda _: True, params.scalers),
    )  # Filter out scaler params

    if config.freeze_lti:
        freeze_mask = eqx.tree_at(
            lambda s: (s[0].A, s[0].B_u, s[0].C_y, s[0].D_yu),
            freeze_mask,
            replace=(True, True, True, True),
        )  # filter out A, B, C, D

    adam_train, adam_val = jnp.empty((0,)), jnp.empty((0,))
    lbfgs_train, lbfgs_val = jnp.empty((0,)), jnp.empty((0,))
    lbfgs_grad_norm = jnp.empty((0,))
    lbfgs_n_iters = 0
    lbfgs_stop_reason = 0
    adam_time, lbfgs_time = 0.0, 0.0

    if config.adam_epochs > 0:
        t0 = time.perf_counter()
        opt_vars, (adam_train, adam_val) = adam_training(
            opt_vars,
            static,
            train_data_scaled,
            config.adam_epochs,
            config.adam_lr,
            loss_fn,
            config.l2_rho,
            val_data_scaled,
            freeze_mask=freeze_mask,
        )
        adam_time = time.perf_counter() - t0

    if config.lbfgs_epochs > 0:
        t0 = time.perf_counter()
        (
            opt_vars,
            (
                lbfgs_train,
                lbfgs_val,
                lbfgs_grad_norm,
                lbfgs_n_iters,
                lbfgs_stop_reason,
            ),
        ) = lbfgs_training(
            opt_vars,
            static,
            train_data_scaled,
            config.lbfgs_epochs,
            loss_fn,
            config.l2_rho,
            config.lbfgs_memory,
            config.lbfgs_max_ls,
            config.lbfgs_warmup,
            val_data_scaled,
            config.grad_tol,
            freeze_mask=freeze_mask,
            ls_tol=config.lbfgs_ls_tol,
            slope_rtol=config.lbfgs_slope_rtol,
            approx_dec_rtol=config.lbfgs_approx_dec_rtol,
            loss_max=config.lbfgs_loss_max,
            stagnation_tol=config.lbfgs_stagnation_tol,
        )
        lbfgs_time = time.perf_counter() - t0
        lbfgs_n_iters = int(lbfgs_n_iters)
        lbfgs_stop_reason = int(lbfgs_stop_reason)

    train = jnp.concatenate([x for x in (adam_train, lbfgs_train) if x.size > 0])
    _val_parts = [x for x in (adam_val, lbfgs_val) if x.size > 0]
    val = jnp.concatenate(_val_parts) if _val_parts else jnp.empty((0,))

    history = OptimHistory(
        train=train,
        val=val,
        adam_train=adam_train,
        lbfgs_train=lbfgs_train,
        adam_val=adam_val,
        lbfgs_val=lbfgs_val,
        adam_time=adam_time,
        lbfgs_time=lbfgs_time,
        config=config,
        timestamp=timestamp,
        lbfgs_grad_norm=lbfgs_grad_norm,
        lbfgs_n_iters=lbfgs_n_iters,
        lbfgs_stop_reason=lbfgs_stop_reason,
    )

    params, x0 = opt_vars
    if is_collection and config.shooting_type != ShootingType.ENCODER:
        x0 = _split_x0(x0, counts, config.shooting_type)

    # Combine model
    sys = eqx.combine(params, static)

    return sys, x0, history


def optimize_ensemble(
    ensemble,
    train_data,
    x0,
    config: OptimConfig,
    *,
    val_data: iddata | iddata_collection | Sequence[iddata] | None = None,
    ensemble_batch_size: int | None = None,
):
    """Optimizes all E models of an ensemble in parallel.

    Works like ``optimize``, but each member has its own parameters and
    initial states, and all members are trained on the same data. The Adam
    phase is vmapped over all members at once. The LBFGS phase is run in
    chunks of ``ensemble_batch_size`` members, sharded across the available
    devices.

    Parameters
    ----------
    ensemble : AbstractEnsemble
        Ensemble to train, e.g. ``EnsembleLinStateSpace``.
    train_data : iddata | iddata_collection | Sequence[iddata]
        Training data in physical units, shared by all members.
    x0 : Array | Sequence[Array]
        Initial guess for the initial state(s). Accepts the forms listed in
        ``optimize``, shared by all members, or per-member states with a
        leading axis E: (E, nx) for single shooting or (E, N_rec, nx) for
        multiple shooting.
    config : OptimConfig
        Training settings, shared by all members.
    val_data : iddata | iddata_collection | Sequence[iddata] | None
        Validation data in physical units. When given, each member's LBFGS
        phase returns its parameters with the lowest validation loss.
    ensemble_batch_size : int | None
        Number of members per LBFGS chunk. Defaults to the number of JAX
        devices. Larger values use more memory but need fewer chunks.

    Returns
    -------
    ensemble : AbstractEnsemble
        Trained ensemble.
    x0 : Array | list[Array]
        Learned initial states, of shape (E, nx) or
        (E, N_rec, nx). For multiple experiments, a list with one entry of
        shape (E, nx) or (E, N_rec_i, nx) per experiment.
    history : OptimHistory
        Losses per member: the loss arrays have shape (n_epochs, E), and
        ``lbfgs_n_iters`` and ``lbfgs_stop_reason`` have shape (E,).
    """
    timestamp = datetime.now().isoformat()

    is_collection = not isinstance(train_data, iddata)
    if is_collection:
        train_data = as_collection(train_data)

    # Initialize scalers using training data, if necessary
    if not ensemble.scalers.is_initialized:
        warnings.warn(
            "DataScaler not yet initialized, initializing using training data.",
            stacklevel=2,
        )
        ensemble = ensemble.with_scalers(ensemble.scalers.initialize(train_data))

    # Scale data for training
    train_data_scaled = ensemble.scalers.transform(train_data)
    val_data_scaled = _scale_validation_iddata(ensemble.scalers, val_data)

    train_data_scaled, counts = _create_training_iddata(
        train_data_scaled, config, getattr(ensemble.models, "n_hist", None)
    )

    loss_fn = multiple_shooting_loss_fn
    if config.shooting_type == ShootingType.ENCODER:
        loss_fn = encoder_loss_fn_ensemble

    # Filter out non-arrays parameters
    filter_spec = jax.tree_util.tree_map(
        eqx.is_array, ensemble
    )  # filter out all non-arrays

    # Split model into trainable and static
    params, static = eqx.partition(ensemble, filter_spec)

    E = ensemble.models.A.shape[0]  # number of ensemble members
    # Broadcast the initial state to the ensemble multiple-shooting shape
    # (E, N_rec, nx).  A single state (nx,) is shared across all records and
    # members; a per-record state (N_rec, nx) is shared across members; an
    # (E, N_rec, nx) array (e.g. learned from a previous run) is used as-is.
    # A list per experiment is first stacked along the record axis.
    if config.shooting_type != ShootingType.ENCODER and counts is not None:
        N_rec = sum(counts)
        x0 = _stack_x0(x0, config.shooting_type)
        if x0.ndim == 2 and x0.shape[0] == 1 and N_rec != 1:
            x0 = x0[0]  # shared state given as (1, nx)
        if x0.ndim == 3:
            if x0.shape[0] != E or x0.shape[1] != N_rec:
                raise ValueError(
                    f"x0 has shape {x0.shape}, expected ({E}, {N_rec}, nx)"
                )
            x0_batched = x0
        elif x0.ndim == 2:
            if x0.shape[0] != N_rec:
                raise ValueError(f"x0 has shape {x0.shape}, expected ({N_rec}, nx)")
            x0_batched = jnp.repeat(x0[None, :], E, axis=0)
        else:
            x0_batched = jnp.repeat(
                jnp.repeat(x0[None, :], N_rec, axis=0)[None, :], E, axis=0
            )
    else:
        # Single or encoder shooting: a single state (nx,) is shared across
        # all ensemble members; a per-member state (E, nx) (e.g. learned
        # from a previous run) is used as-is.  For encoder shooting x0 is
        # ignored (regenerated by StateEncoder), so this is safe either way.
        if x0.ndim == 2 and x0.shape[0] == E:
            x0_batched = x0
        else:
            x0_batched = jnp.repeat(_squeeze_single_state(x0)[None, :], E, axis=0)

    opt_vars = (params, x0_batched)

    # Create mask for freezing parameters
    freeze_mask = jax.tree_util.tree_map(lambda _: False, opt_vars)
    freeze_mask = eqx.tree_at(
        lambda s: s[0].models.scalers,
        freeze_mask,
        jax.tree.map(lambda _: True, params.models.scalers),
    )  # Filter out scaler params

    if config.freeze_lti:
        freeze_mask = eqx.tree_at(
            lambda s: (
                s[0].models.A,
                s[0].models.B_u,
                s[0].models.C_y,
                s[0].models.D_yu,
            ),
            freeze_mask,
            replace=(True, True, True, True),
        )  # filter out LTI params

    #    Create mesh
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=("ensemble",))

    sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("ensemble"))
    opt_vars = jax.device_put(opt_vars, sharding)

    adam_train, adam_val = jnp.empty((0,)), jnp.empty((0,))
    lbfgs_train, lbfgs_val = jnp.empty((0,)), jnp.empty((0,))
    lbfgs_grad_norm = jnp.empty((0,))
    lbfgs_n_iters = jnp.empty((0,), dtype=jnp.int32)
    lbfgs_stop_reason = jnp.empty((0,), dtype=jnp.int32)
    adam_time, lbfgs_time = 0.0, 0.0

    if config.adam_epochs > 0:
        t0 = time.perf_counter()
        opt_vars, (adam_train, adam_val) = adam_training_ensemble(
            opt_vars,
            static,
            train_data_scaled,
            config.adam_epochs,
            config.adam_lr,
            loss_fn,
            config.l2_rho,
            val_data=val_data_scaled,
            freeze_mask=freeze_mask,
        )
        # The ensemble scan returns before the work is done, wait so the timer
        # covers it.
        jax.block_until_ready((opt_vars, adam_train, adam_val))
        adam_time = time.perf_counter() - t0

    if config.lbfgs_epochs > 0:
        batched_lbfgs = eqx.filter_jit(
            eqx.filter_vmap(
                lbfgs_training,
                in_axes=(
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
        )

        n_devices = len(jax.devices())
        batch_size = ensemble_batch_size or n_devices
        n_chunks = (E + batch_size - 1) // batch_size
        padded_E = n_chunks * batch_size
        pad = padded_E - E

        # Pad to a multiple of batch_size by repeating the first ``pad``
        # models so that every chunk has the same size.  This lets JAX
        # compile a single vmapped program and reuse it for all chunks.
        if pad > 0:
            pad_vars = jax.tree_util.tree_map(lambda x: x[:pad], opt_vars)
            opt_vars = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0), opt_vars, pad_vars
            )

        chunk_params_list = []
        chunk_train_list = []
        chunk_val_list = []
        chunk_grad_list = []
        chunk_n_iters_list = []
        chunk_stop_list = []

        t0 = time.perf_counter()
        for chunk_idx in range(n_chunks):
            start = chunk_idx * batch_size
            stop = start + batch_size
            chunk_vars = jax.tree_util.tree_map(lambda x: x[start:stop], opt_vars)
            chunk_vars = jax.device_put(chunk_vars, sharding)

            (
                chunk_params,
                (
                    chunk_train,
                    chunk_val,
                    chunk_grad,
                    chunk_n_iters,
                    chunk_stop,
                ),
            ) = batched_lbfgs(
                chunk_vars,
                static,
                train_data_scaled,
                config.lbfgs_epochs,
                loss_fn,
                config.l2_rho,
                config.lbfgs_memory,
                config.lbfgs_max_ls,
                config.lbfgs_warmup,
                val_data_scaled,
                config.grad_tol,
                freeze_mask=freeze_mask,
                ls_tol=config.lbfgs_ls_tol,
                slope_rtol=config.lbfgs_slope_rtol,
                approx_dec_rtol=config.lbfgs_approx_dec_rtol,
                loss_max=config.lbfgs_loss_max,
                stagnation_tol=config.lbfgs_stagnation_tol,
            )
            jax.block_until_ready((chunk_params, chunk_train, chunk_val))

            chunk_params_list.append(chunk_params)
            chunk_train_list.append(chunk_train)
            chunk_val_list.append(chunk_val)
            chunk_grad_list.append(chunk_grad)
            chunk_n_iters_list.append(chunk_n_iters)
            chunk_stop_list.append(chunk_stop)

        lbfgs_time = time.perf_counter() - t0

        # Reassemble chunks and strip padding
        opt_vars = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(list(xs), axis=0), *chunk_params_list
        )
        lbfgs_train = jnp.concatenate(chunk_train_list, axis=0)
        lbfgs_val = jnp.concatenate(chunk_val_list, axis=0)
        lbfgs_grad_norm = jnp.concatenate(chunk_grad_list, axis=0)
        lbfgs_n_iters = jnp.concatenate(chunk_n_iters_list, axis=0)
        lbfgs_stop_reason = jnp.concatenate(chunk_stop_list, axis=0)

        if pad > 0:
            opt_vars = jax.tree_util.tree_map(lambda x: x[:E], opt_vars)
            lbfgs_train = lbfgs_train[:E]
            lbfgs_val = lbfgs_val[:E]
            lbfgs_grad_norm = lbfgs_grad_norm[:E]
            lbfgs_n_iters = lbfgs_n_iters[:E]
            lbfgs_stop_reason = lbfgs_stop_reason[:E]

    # In the ensemble path Adam losses are (steps, E) from scan, while LBFGS
    # losses are (E, steps) from filter_vmap; transpose LBFGS to align them.
    if lbfgs_train.size > 0:
        lbfgs_train = jnp.swapaxes(lbfgs_train, 0, 1)
    if lbfgs_val.size > 0:
        lbfgs_val = jnp.swapaxes(lbfgs_val, 0, 1)
    if lbfgs_grad_norm.size > 0:
        lbfgs_grad_norm = jnp.swapaxes(lbfgs_grad_norm, 0, 1)

    train = jnp.concatenate(
        [x for x in (adam_train, lbfgs_train) if x.size > 0], axis=0
    )
    _val_parts = [x for x in (adam_val, lbfgs_val) if x.size > 0]
    val = jnp.concatenate(_val_parts, axis=0) if _val_parts else jnp.empty((0,))

    history = OptimHistory(
        train=train,
        val=val,
        adam_train=adam_train,
        lbfgs_train=lbfgs_train,
        adam_val=adam_val,
        lbfgs_val=lbfgs_val,
        adam_time=adam_time,
        lbfgs_time=lbfgs_time,
        config=config,
        timestamp=timestamp,
        lbfgs_grad_norm=lbfgs_grad_norm,
        lbfgs_n_iters=lbfgs_n_iters,
        lbfgs_stop_reason=lbfgs_stop_reason,
    )

    params, x0 = opt_vars
    if is_collection and config.shooting_type != ShootingType.ENCODER:
        x0 = _split_x0(x0, counts, config.shooting_type)

    # Combine model
    ensemble = eqx.combine(params, static)

    return ensemble, x0, history


def learn_initial_state(
    sys: eqx.Module,
    train_data: iddata | iddata_collection | Sequence[iddata],
    x0: Array | Sequence[Array],
    lbfgs_epochs: int,
    T: int | None = None,
    l2_rho: float = 0.0,
):
    """Learns initial state by minimizing the mean-squared-simulation error on the (truncated) provided data-set. If T is set only the first T timesteps of the data-set are used.

    Parameters
    ----------
    sys : eqx.Module
        model of dynamic system used for simulation
    train_data : iddata | iddata_collection | Sequence[iddata]
        iddata object containing trajectories used for optimization, or
        multiple experiments for which an initial state is learned each.
    x0 : Array | Sequence[Array]
        Initial guess of the initial state: a single state of shape (nx,) or
        (1, nx) shared by all experiments, or a list with one such state per
        experiment.
    lbfgs_epochs : int
        Number of LBFGS epochs in training
    T : int | None
        Truncation length.

    Returns
    -------
    x0: Array, (nx, 1) | list[Array]
        Learned initial state, a list with one state per experiment for
        multiple experiments.
    lbfgs_losses: Array
        History of the loss function
    """

    is_collection = not isinstance(train_data, iddata)
    if is_collection:
        train_data = as_collection(train_data)

    train_data_scaled = sys.scalers.transform(train_data)  # add way to truncate data
    if T is not None:
        if is_collection:
            train_data_scaled = iddata_collection(
                [truncate_iddata(d, T) for d in train_data_scaled]
            )
        else:
            train_data_scaled = truncate_iddata(train_data_scaled, T)

    if is_collection:
        counts = [1] * len(train_data_scaled)
        train_data_scaled = create_padded_iddata(train_data_scaled)
        x0 = _broadcast_x0(_stack_x0(x0, ShootingType.SINGLE), len(counts))
    loss_fn = multiple_shooting_loss_fn

    filter_spec = jax.tree_util.tree_map(lambda _: False, sys)
    params, static = eqx.partition(
        sys, filter_spec
    )  # Partitions everything to be static, needed for compatibility with optimizers
    opt_vars = (params, x0)

    freeze_mask = jax.tree_util.tree_map(lambda _: False, opt_vars)

    (params, x0), (lbfgs_losses, *_) = lbfgs_training(
        opt_vars,
        static,
        train_data_scaled,
        lbfgs_epochs,
        loss_fn,
        l2_rho,
        20,
        20,
        False,
        freeze_mask=freeze_mask,
    )

    if is_collection:
        x0 = _split_x0(x0, counts, ShootingType.SINGLE)

    return x0, lbfgs_losses
