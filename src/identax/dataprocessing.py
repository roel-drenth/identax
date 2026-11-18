"""Data containers, scaling and windowing utilities for system identification.

This module provides:

- ``iddata`` and ``iddata_collection``: containers for input (U), output (Y)
  and optional exogenous (D) signals of one or more experiments.
- ``DataScaler`` and ``DataScalerConfig``: per-signal normalization that is
  stored inside a model, so that models operate on scaled data internally
  while users pass and receive data in physical units.
- Helpers that cut trajectories into (overlapping) windows for multiple
  shooting.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class ScalerType(StrEnum):
    """Available scaling methods.

    - ``STANDARD``: subtract the mean and divide by the standard deviation,
      computed per channel over the training samples.
    - ``IDENTITY``: leave the signal unchanged.
    """

    STANDARD = "standard"
    IDENTITY = "identity"


def _as_float_array(x):
    """Convert x to a JAX array of the default float type; None stays None."""
    if x is None:
        return None
    return jnp.asarray(x, dtype=float)


def _validate_iddata_dimensions(
    U: Array, Y: Array, D: Array | None, mask: Array | None = None
):
    if U.shape[:-1] != Y.shape[:-1]:
        raise ValueError(
            f"U and Y must have matching sample dimensions. "
            f"Got {U.shape[:-1]} and {Y.shape[:-1]}"
        )
    if D is not None and U.shape[:-1] != D.shape[:-1]:
        raise ValueError(
            f"U, Y and D must have matching sample dimensions. "
            f"Got {U.shape[:-1]}, {Y.shape[:-1]}  and {D.shape[:-1]}"
        )
    if mask is not None and U.shape[:-1] != mask.shape:
        raise ValueError(
            f"mask must match the sample dimensions of U. "
            f"Got {mask.shape} and {U.shape[:-1]}"
        )


class AbstractScaler(eqx.Module):
    """Interface for scaling a single signal (U, Y or D).

    Subclasses implement a forward and inverse transform, a classmethod
    ``from_data`` that fits the scaler on data of shape (N, n), and a
    classmethod ``from_shape`` that creates an unfitted placeholder.
    """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the scaler has been fitted and can be used."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, x):
        """Map a signal from physical units to scaled units."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x):
        """Map a signal from scaled units back to physical units."""
        raise NotImplementedError

    @abstractmethod
    def from_data(self, x):
        """Return a scaler fitted on the samples in ``x``."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_shape(cls, dim):
        """Return an unfitted scaler for ``dim`` channels.

        Used as a template when loading a saved model.
        """
        raise NotImplementedError


class StandardScaler(AbstractScaler):
    """Per-channel standardization, ``(x - mean) / std``.

    An unfitted scaler stores NaN for mean and std. Use ``from_data`` to fit
    it, or ``from_shape`` to create a placeholder with the right shape (for
    example as a template when deserializing a model).

    Attributes
    ----------
    mean : Array
        Per-channel mean, of shape (1, n).
    std : Array
        Per-channel standard deviation, of shape (1, n).
    """

    mean: Array
    std: Array

    def __init__(
        self, mean: Array = jnp.array([[jnp.nan]]), std: Array = jnp.array([[jnp.nan]])
    ):
        self.mean = mean
        self.std = std

    @property
    def is_initialized(self) -> bool:
        # Check if mean and std are initialized (not all NaN)
        return not (jnp.isnan(self.mean).all() and jnp.isnan(self.std).all())

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_data(cls, x, axis=(0,)):
        """Fit the scaler on data.

        Parameters
        ----------
        x : Array
            Samples of shape (N, n).
        axis : tuple[int, ...]
            Axes over which mean and std are computed. Defaults to the
            sample axis.
        """
        mean = jnp.mean(x, axis=axis, keepdims=True)

        std = jnp.std(x, axis=axis, keepdims=True)
        return cls(mean=mean, std=std)

    @classmethod
    def from_shape(cls, dim):
        """Create an unfitted scaler (NaN mean and std) for ``dim`` channels."""
        mean = jnp.full((1, dim), jnp.nan)
        std = jnp.full((1, dim), jnp.nan)
        return cls(mean=mean, std=std)


class IdentityScaler(AbstractScaler):
    """Scaler that leaves the signal unchanged. It needs no fitting."""

    @property
    def is_initialized(self) -> bool:
        return True

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    @classmethod
    def from_data(cls, x, axis=(0,)):
        """Return an ``IdentityScaler``; the data is ignored."""
        return cls()

    @classmethod
    def from_shape(cls, dim):
        """Return an ``IdentityScaler``; it has no parameters to size."""
        return cls()


@dataclass(frozen=True, kw_only=True)
class DataScalerConfig:
    """Choice of scaling method for each signal.

    Parameters
    ----------
    type_u : ScalerType | str
        Scaler for the input U. Default ``"standard"``.
    type_y : ScalerType | str
        Scaler for the output Y. Default ``"standard"``.
    type_d : ScalerType | str | None
        Scaler for the exogenous signal D. Must be set for models with
        ``nd > 0``; ``None`` means the model uses no D.
    """

    type_u: ScalerType | str = ScalerType.STANDARD
    type_y: ScalerType | str = ScalerType.STANDARD
    type_d: ScalerType | str | None = None

    def __post_init__(self):
        if isinstance(self.type_u, str):
            object.__setattr__(self, "type_u", ScalerType(self.type_u))
        if isinstance(self.type_y, str):
            object.__setattr__(self, "type_y", ScalerType(self.type_y))
        if isinstance(self.type_d, str):
            object.__setattr__(self, "type_d", ScalerType(self.type_d))

    @classmethod
    def from_spec(
        cls, spec: DataScalerConfig | ScalerType | str | dict, nd: int = 0
    ) -> DataScalerConfig:
        """Build a config from a shorthand specification.

        Parameters
        ----------
        spec : DataScalerConfig | ScalerType | str | dict
            Either a config (returned unchanged), a single scaler type that
            is applied to all signals, or a dict of keyword arguments for
            ``DataScalerConfig``.
        nd : int
            Number of exogenous signals. When ``nd > 0``, a single scaler
            type is also applied to D, and a given config must set
            ``type_d``.

        Returns
        -------
        DataScalerConfig
        """
        if isinstance(spec, cls):
            if nd > 0 and spec.type_d is None:
                raise ValueError("type_d must be set when nd > 0")
            return spec
        if isinstance(spec, str):
            spec = ScalerType(spec)
        if isinstance(spec, dict):
            return cls(**spec)
        if isinstance(spec, ScalerType):
            return cls(type_u=spec, type_y=spec, type_d=spec if nd > 0 else None)
        raise TypeError(f"Unexpected spec type: {type(spec)}")


class DataScaler(eqx.Module):
    """Scalers for the U, Y and (optional) D signals of a model.

    Every model holds a ``DataScaler``. It is created unfitted from the
    model's ``DataScalerConfig`` and fitted on the training data on the first
    call to ``optimize`` (see ``initialize_scalers``).

    Parameters
    ----------
    config : DataScalerConfig | None
        Creates unfitted scalers of the configured types.
    u_scaler, y_scaler, d_scaler : AbstractScaler | None
        Ready-made scaler instances. Cannot be combined with ``config``.
    """

    u_scaler: AbstractScaler | None
    y_scaler: AbstractScaler | None
    d_scaler: AbstractScaler | None

    def __init__(
        self,
        config: DataScalerConfig | None = None,
        *,
        u_scaler: AbstractScaler | None = None,
        y_scaler: AbstractScaler | None = None,
        d_scaler: AbstractScaler | None = None,
    ):
        if config is not None and (
            u_scaler is not None or y_scaler is not None or d_scaler is not None
        ):
            raise ValueError(
                "Cannot specify both config and individual scaler arguments"
            )
        if config is not None:
            self.u_scaler = SCALERS[config.type_u]()
            self.y_scaler = SCALERS[config.type_y]()
            self.d_scaler = (
                SCALERS[config.type_d]() if (config.type_d is not None) else None
            )
        else:
            self.u_scaler = u_scaler
            self.y_scaler = y_scaler
            self.d_scaler = d_scaler

    @property
    def is_initialized(self) -> bool:
        """Whether all configured scalers have been fitted."""
        if self.d_scaler is not None:
            return (
                self.u_scaler.is_initialized
                and self.y_scaler.is_initialized
                and self.d_scaler.is_initialized
            )
        return self.u_scaler.is_initialized and self.y_scaler.is_initialized

    def transform(self, data: iddata | iddata_collection):
        """Scale data from physical units to model units.

        D is scaled only when a ``d_scaler`` is configured (``nd > 0``).
        Without one, D is dropped, so models with ``nd = 0`` ignore it.
        History signals (``U_hist``, ...) are not carried over; the mask is.

        Parameters
        ----------
        data : iddata | iddata_collection
            Data in physical units.

        Returns
        -------
        iddata | iddata_collection
            Scaled data, of the same type as ``data``.

        Raises
        ------
        ValueError
            If a ``d_scaler`` is configured but ``data`` contains no D.
        """
        if isinstance(data, iddata_collection):
            return iddata_collection([self.transform(d) for d in data])

        u_scaled = self.u_scaler.transform(data.U)
        y_scaled = self.y_scaler.transform(data.Y)

        if self.d_scaler is not None:
            if not data.use_exo:
                raise ValueError(
                    "Model expects exogenous signals (nd > 0), but data contains no D."
                )
            d_scaled = self.d_scaler.transform(data.D)
            return iddata(u_scaled, y_scaled, D=d_scaled, mask=data.mask)

        return iddata(u_scaled, y_scaled, mask=data.mask)

    def inverse_transform(self, data_scaled: iddata | iddata_collection):
        """Map scaled data back to physical units; the inverse of ``transform``."""
        if isinstance(data_scaled, iddata_collection):
            return iddata_collection([self.inverse_transform(d) for d in data_scaled])

        u = self.u_scaler.inverse_transform(data_scaled.U)
        y = self.y_scaler.inverse_transform(data_scaled.Y)

        if self.d_scaler is not None and data_scaled.use_exo:
            d = self.d_scaler.inverse_transform(data_scaled.D)
            return iddata(u, y, D=d, mask=data_scaled.mask)

        return iddata(u, y, mask=data_scaled.mask)

    def initialize(self, data: iddata | iddata_collection):
        """Return a new ``DataScaler`` of the same types, fitted on ``data``.

        For a collection, the scalers are fitted on the samples of all
        experiments pooled together.
        """
        if isinstance(data, iddata_collection):
            data = concatenate_iddata(data.experiments)

        if self.d_scaler is not None and not data.use_exo:
            raise ValueError(
                "Model expects exogenous signals (nd > 0), but data contains no D."
            )

        return DataScaler(
            u_scaler=type(self.u_scaler).from_data(data.U),
            y_scaler=type(self.y_scaler).from_data(data.Y),
            d_scaler=(
                type(self.d_scaler).from_data(data.D)
                if self.d_scaler is not None
                else None
            ),
        )

    def init_empty(self, nu: int, ny: int, nd: int):
        """Return unfitted scalers of the same types, sized for nu, ny and nd."""
        return DataScaler(
            u_scaler=type(self.u_scaler).from_shape(nu),
            y_scaler=type(self.y_scaler).from_shape(ny),
            d_scaler=(
                type(self.d_scaler).from_shape(nd)
                if self.d_scaler is not None
                else None
            ),
        )

    @classmethod
    def from_data(cls, data: iddata, config: DataScalerConfig):
        """Create scalers of the configured types, fitted on ``data``."""
        return cls(config).initialize(data)

    @classmethod
    def from_shape(cls, config: DataScalerConfig, nu: int, ny: int, nd: int = 0):
        """Create unfitted scalers of the configured types and given sizes."""
        return cls(config).init_empty(nu, ny, nd)


def initialize_scalers(
    model: eqx.Module, data: iddata | iddata_collection
) -> eqx.Module:
    """Fit the model's scalers on ``data`` if they are not fitted yet.

    Parameters
    ----------
    model : eqx.Module
        Model with a ``scalers`` attribute of type ``DataScaler``.
    data : iddata | iddata_collection
        Training data in physical units.

    Returns
    -------
    eqx.Module
        The model with fitted scalers. Returned unchanged if its scalers were
        already fitted.
    """
    if model.scalers.is_initialized:
        return model

    return eqx.tree_at(
        lambda m: (m.scalers,),
        model,
        (model.scalers.initialize(data),),
    )


def create_multiple_shooting_data(
    X: Array,
    T: int,
    overlap: int,
) -> tuple[Array, int]:
    """Cut a trajectory into sub-records of length T for multiple shooting.

    Consecutive sub-records start ``T - overlap`` samples apart. Trailing samples
    that do not fill a complete window are discarded.

    Parameters
    ----------
    X : Array
        Trajectory of shape (N, n).
    T : int
        Window length.
    overlap : int
        Number of samples shared by consecutive windows (``0 <= overlap < T``).

    Returns
    -------
    Xms : Array
        Windows of shape (num_records, T, n).
    num_records : int
        Number of windows.
    """
    N = X.shape[0]
    step_size = T - overlap

    num_records = (N - T) // step_size + 1
    idx_start = jnp.arange(num_records) * (step_size)

    idx = idx_start[:, None] + jnp.arange(T)[None, :]

    Xms = X[idx]

    return Xms, num_records


def create_encoder_data(
    X: Array, T: int, n_hist: int, overlap: int
) -> tuple[Array, Array, int]:
    """Cut a trajectory into sub-records with a preceding history, for encoder training.

    Each sub-record of length T is paired with the ``n_hist`` samples directly
    before it, which a state encoder uses to estimate the window's initial
    state. Consecutive windows start ``T - overlap`` samples apart.

    Parameters
    ----------
    X : Array
        Trajectory of shape (N, n).
    T : int
        Window length.
    n_hist : int
        Number of history samples before each window.
    overlap : int
        Number of samples shared by consecutive windows (``0 <= overlap < T``).

    Returns
    -------
    Xms : Array
        Windows of shape (num_records, T, n).
    Xhist : Array
        Histories of shape (num_records, n_hist, n).
    num_records : int
        Number of windows.
    """
    N = X.shape[0]
    nx = X.shape[1]
    step_size = T - overlap

    num_records = (N - T - n_hist) // step_size + 1
    idx_start_hist = jnp.arange(num_records) * (step_size)
    idx_start = idx_start_hist + n_hist

    Xhist = jax.vmap(
        lambda s: jax.lax.dynamic_slice(
            X,
            (s, 0),
            (n_hist, nx),
        )
    )(idx_start_hist)

    Xms = jax.vmap(
        lambda s: jax.lax.dynamic_slice(
            X,
            (s, 0),
            (T, nx),
        )
    )(idx_start)

    return Xms, Xhist, num_records


class iddata(eqx.Module):
    """Input-output data of a single experiment.

    Signals have shape (N, n) for a full trajectory. Internally, the same
    container also holds windowed data of shape (R, T, n) or padded
    experiments of shape (n_exp, N_max, n).

    Parameters
    ----------
    U : Array
        Input signal, of shape (N, nu).
    Y : Array
        Output signal, of shape (N, ny).
    D : Array | None
        Exogenous signal, of shape (N, nd). Used only by models with
        ``nd > 0``.
    U_hist, Y_hist, D_hist : Array | None
        Past samples preceding each window, used by the state encoder.
        Normally created by the optimizer, not by the user.
    mask : Array | None
        Boolean array over the sample dimensions that marks valid
        (non-padded) samples. ``None`` means all samples are valid.

    Attributes
    ----------
    use_exo : bool
        Whether D is present.

    Raises
    ------
    ValueError
        If the sample dimensions of U, Y, D and mask do not match.
    """

    U: Array
    Y: Array
    D: Array | None = None
    U_hist: Array | None = None
    Y_hist: Array | None = None
    D_hist: Array | None = None
    mask: Array | None = None
    use_exo: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        U: Array,
        Y: Array,
        *,
        D: Array | None = None,
        U_hist: Array | None = None,
        Y_hist: Array | None = None,
        D_hist: Array | None = None,
        mask: Array | None = None,
    ):
        # Signals are stored as JAX arrays of the default float type (float64,
        # as identax enables x64), so that e.g. float32 numpy data does not
        # lead to float32 scalers or mixed-precision training.
        U, Y, D, U_hist, Y_hist, D_hist = (
            _as_float_array(x) for x in (U, Y, D, U_hist, Y_hist, D_hist)
        )
        _validate_iddata_dimensions(U, Y, D, mask)
        self.U = U
        self.Y = Y

        self.U_hist = U_hist
        self.Y_hist = Y_hist
        # mask marks valid (non-padded) samples, None means all samples are valid
        self.mask = mask
        if D is not None:
            self.use_exo = True
            self.D = D
            self.D_hist = D_hist
        else:
            self.use_exo = False


@dataclass(frozen=True)
class iddata_collection:
    """Collection of experiments, each stored as an iddata with its own length N.

    Parameters
    ----------
    experiments : Sequence[iddata]
        One iddata per experiment, with U of shape (N_i, nu), Y of shape
        (N_i, ny) and, optionally, D of shape (N_i, nd).
    """

    experiments: tuple[iddata, ...]

    def __post_init__(self):
        experiments = tuple(self.experiments)
        object.__setattr__(self, "experiments", experiments)

        if len(experiments) == 0:
            raise ValueError("iddata_collection requires at least one experiment.")

        ref = experiments[0]
        for i, data in enumerate(experiments):
            if not isinstance(data, iddata):
                raise TypeError(f"Experiment {i} is not an iddata, got {type(data)}.")
            if data.U.ndim != 2:
                raise ValueError(
                    f"Experiment {i} must have U of shape (N, nu), got {data.U.shape}."
                )
            if data.use_exo != ref.use_exo:
                raise ValueError(
                    "Either all experiments or none of them must contain D."
                )
            if (
                data.U.shape[-1] != ref.U.shape[-1]
                or data.Y.shape[-1] != ref.Y.shape[-1]
                or (data.use_exo and data.D.shape[-1] != ref.D.shape[-1])
            ):
                raise ValueError(
                    f"Experiment {i} has different signal dimensions than experiment 0."
                )

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        return iter(self.experiments)


def as_collection(data: iddata_collection | Sequence[iddata]) -> iddata_collection:
    """Return ``data`` as an ``iddata_collection``, wrapping a sequence if needed."""
    if isinstance(data, iddata_collection):
        return data
    return iddata_collection(data)


def concatenate_iddata(datas: Sequence[iddata]) -> iddata:
    """Concatenates iddata objects along the first axis, i.e. time for
    trajectories of shape (N, n) and records for windows of shape (R, T, n).
    """

    def concat(arrays):
        if arrays[0] is None:
            return None
        return jnp.concatenate(arrays, axis=0)

    return iddata(
        concat([d.U for d in datas]),
        concat([d.Y for d in datas]),
        D=concat([d.D for d in datas]),
        U_hist=concat([d.U_hist for d in datas]),
        Y_hist=concat([d.Y_hist for d in datas]),
        D_hist=concat([d.D_hist for d in datas]),
    )


def create_padded_iddata(data: iddata_collection) -> iddata:
    """Pads all experiments to the longest length and stacks them into an
    iddata of shape (n_exp, N_max, n). Padded samples are zero, which for
    scaled data corresponds to the mean of the training data, and mask marks
    the valid samples (None if all experiments have equal length).
    """
    lengths = [d.U.shape[0] for d in data]
    N_max = max(lengths)

    def pad_and_stack(arrays):
        if arrays[0] is None:
            return None
        return jnp.stack(
            [jnp.pad(X, ((0, N_max - X.shape[0]), (0, 0))) for X in arrays]
        )

    mask = None
    if min(lengths) != N_max:
        mask = jnp.arange(N_max)[None, :] < jnp.array(lengths)[:, None]

    return iddata(
        pad_and_stack([d.U for d in data]),
        pad_and_stack([d.Y for d in data]),
        D=pad_and_stack([d.D for d in data]),
        mask=mask,
    )


def simulate_collection(
    model: eqx.Module,
    data: iddata_collection | Sequence[iddata],
    x0: Array | Sequence[Array],
) -> list[tuple[Array, Array]]:
    """Simulates the model for each experiment in the collection.

    Parameters
    ----------
    model   : eqx.Module
        Model or ensemble with a simulate(data, x0) method.
    data    : iddata_collection | Sequence[iddata]
        Experiments to simulate.
    x0      : Array | Sequence[Array]
        Initial state shared by all experiments, or one initial state per
        experiment (e.g. as returned by optimize).

    Returns
    -------
    list[tuple[Array, Array]]
        Output of model.simulate, i.e. (y_accum, x_accum), per experiment.
    """
    data = as_collection(data)
    if not isinstance(x0, (list, tuple)):
        x0 = [x0] * len(data)
    if len(x0) != len(data):
        raise ValueError(f"Got {len(x0)} initial states for {len(data)} experiments.")
    return [model.simulate(d, x) for d, x in zip(data, x0)]


def create_multiple_shooting_iddata(
    data: iddata, T: int, overlap: int
) -> tuple[iddata, int]:
    """Cut U, Y and (if present) D into multiple-shooting windows.

    See ``create_multiple_shooting_data`` for the windowing. The mask and
    history signals of ``data`` are not carried over.

    Returns
    -------
    iddata_ms : iddata
        Windowed data, with signals of shape (num_records, T, n).
    num_records : int
        Number of windows.
    """
    U_ms, num_records = create_multiple_shooting_data(data.U, T, overlap)
    Y_ms, _ = create_multiple_shooting_data(data.Y, T, overlap)
    D_ms = None
    if data.use_exo:
        D_ms, _ = create_multiple_shooting_data(data.D, T, overlap)

    iddata_ms = iddata(U_ms, Y_ms, D=D_ms)
    return iddata_ms, num_records


def truncate_iddata(data: iddata, T: int) -> iddata:
    """Keep the first T samples of U, Y and (if present) D.

    The mask and history signals of ``data`` are not carried over.
    """
    U_trunc = data.U[:T]
    Y_trunc = data.Y[:T]
    D_trunc = None
    if data.use_exo:
        D_trunc = data.D[:T]
    return iddata(U_trunc, Y_trunc, D=D_trunc)


def create_encoder_iddata(
    data: iddata, T: int, n_hist: int, overlap: int
) -> tuple[iddata, int]:
    """Cut U, Y and (if present) D into windows with histories, for encoder training.

    See ``create_encoder_data`` for the windowing. The histories are stored
    in ``U_hist``, ``Y_hist`` and ``D_hist``.

    Returns
    -------
    iddata_enc : iddata
        Windowed data, with signals of shape (num_records, T, n) and
        histories of shape (num_records, n_hist, n).
    num_records : int
        Number of windows.
    """
    U_ms, U_hist, num_records = create_encoder_data(data.U, T, n_hist, overlap)
    Y_ms, Y_hist, _ = create_encoder_data(data.Y, T, n_hist, overlap)
    D_ms = None
    D_hist = None
    if data.use_exo:
        D_ms, D_hist, _ = create_encoder_data(data.D, T, n_hist, overlap)

    iddata_enc = iddata(U_ms, Y_ms, D=D_ms, U_hist=U_hist, Y_hist=Y_hist, D_hist=D_hist)
    return iddata_enc, num_records


SCALERS = {
    ScalerType.STANDARD: StandardScaler,
    ScalerType.IDENTITY: IdentityScaler,
}
