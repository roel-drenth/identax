"""Model classes, their configurations and ensembles.

All models are discrete-time state-space models. The LFR models connect an
LTI system to a static block in feedback::

    x[k+1] = A x[k] + B_w w[k] + B_u u[k]
    z[k]   = C_z x[k] + D_zw w[k] + D_zu u[k]
    y[k]   = C_y x[k] + D_yw w[k] + D_yu u[k]

``D_zw`` is zero in all models except the rational LPV-LFR model.

Available models, each with a config class and an ensemble version:

- ``ModelLinStateSpace`` (``LinStateSpaceConfig``): LTI model.
- ``ModelAffineLPVLFR`` (``AffineLPVLFRConfig``): LPV-LFR model with affine
  scheduling dependency, ``w = Delta(p) z``, where ``Delta(p)`` is diagonal
  and repeats each scheduling variable ``nreps`` times.
- ``ModelRationalLPVLFR`` (``RationalLPVLFRConfig``): LPV-LFR model with
  rational scheduling dependency (``D_zw`` nonzero), using a direct
  parameterization of ``D_zw`` that guarantees well-posedness.
- ``ModelNLLFR`` (``NLLFRConfig``): NL-LFR model, ``w = Delta(z)`` with a
  neural network Delta and ``D_zw = 0``.

In the LPV models, the scheduling variable ``p = psi(x, u, d)`` is given by
a neural network scheduling map. Models with a ``StateEncoder`` can be
trained with encoder shooting.

Models store their own ``DataScaler``: ``simulate`` takes and returns data
in physical units, while the internal ``_simulate`` works on scaled data.

Ensembles (``Ensemble*``) hold ``n_models`` models of the same type as one
vmapped module, for parallel training with ``optimize_ensemble``.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray

from ._model_init import init_lfr_params, init_lti_params, scale_nllfr
from .dataprocessing import (
    DataScaler,
    DataScalerConfig,
    ScalerType,
    as_collection,
    create_encoder_iddata,
    iddata,
    iddata_collection,
    initialize_scalers,
)
from .networks import (
    ACTIVATIONS,
    NETWORKS,
    AbstractSchedulingMap,
    NetworkConfig,
    NNSchedulingMap,
    NNStateEncoder,
    SchedulingMapConfig,
    StateEncoderConfig,
)

KEY = 42


def _validate_nreps(npar: int, nreps: int | tuple[int, ...]) -> None:
    """Check that nreps is a positive int or a tuple of npar positive ints.

    Parameters
    ----------
    npar    : int
        Number of scheduling variables.
    nreps   : int | tuple[int, ...]
        Number of repetitions of each scheduling variable.

    Raises
    ------
    ValueError
        If nreps is not positive, or is a tuple whose length differs from npar.
    TypeError
        If nreps is not an int or a tuple.
    """
    if isinstance(nreps, int):
        if nreps <= 0:
            raise ValueError(f"nreps must be > 0, got {nreps}")
        return

    if isinstance(nreps, tuple):
        if len(nreps) != npar:
            raise ValueError(f"Expected {npar} repetitions, got {len(nreps)}")

        if any(rep <= 0 for rep in nreps):
            raise ValueError("All entries of nreps must be positive")
        return

    raise TypeError("nreps must be a positive int or a tuple of positive ints")


def _solve_3x3_cramer(M: Array, b: Array) -> Array:
    """Solve M z = b for a 3x3 system via Cramer's rule (closed-form inverse).

    Avoids the LAPACK-backed ``jnp.linalg.solve`` so the linear solve is
    fully fusible by XLA. ``M`` is (3, 3), ``b`` is (3,); returns ``z`` of
    shape (3,).
    """
    a, b0, c = M[0, 0], M[0, 1], M[0, 2]
    d, e, f = M[1, 0], M[1, 1], M[1, 2]
    g, h, i = M[2, 0], M[2, 1], M[2, 2]
    r0, r1, r2 = b[0], b[1], b[2]

    det = a * (e * i - f * h) - b0 * (d * i - f * g) + c * (d * h - e * g)

    z0 = ((e * i - f * h) * r0 + (c * h - b0 * i) * r1 + (b0 * f - c * e) * r2) / det
    z1 = ((f * g - d * i) * r0 + (a * i - c * g) * r1 + (c * d - a * f) * r2) / det
    z2 = ((d * h - e * g) * r0 + (b0 * g - a * h) * r1 + (a * e - b0 * d) * r2) / det

    return jnp.stack([z0, z1, z2])


@dataclass(frozen=True, kw_only=True)
class LinStateSpaceConfig:
    """Configuration for linear state-space model

    Parameters
    ----------
    nx      : int
            Number of states.
    nu      : int
            Number of inputs.
    ny      : int
            Number of outputs.
    scalers : DataScalerConfig | ScalerType | str
            Configuration of the signal scaling strategies.

    """

    nx: int
    nu: int
    ny: int
    scalers: DataScalerConfig | ScalerType | str = ScalerType.STANDARD

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError(f"nx must be > 0, got {self.nx}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.ny <= 0:
            raise ValueError(f"ny must be > 0, got {self.ny}")

        # Convert scalers to DataScalerConfig (handles string, ScalerType, dict, or DataScalerConfig)
        object.__setattr__(
            self, "scalers", DataScalerConfig.from_spec(self.scalers, nd=0)
        )


@dataclass(frozen=True, kw_only=True)
class AffineLPVLFRConfig:
    """Configuration for LPV models with affine-dependency linear fractional representation

    Parameters
    ----------
    nx      : int
            Number of states.
    nu      : int
            Number of inputs.
    ny      : int
            Number of outputs.
    npar    : int
            Number of scheduling variables.
    nreps   : int | tuple
            Number of repetitions of each scheduling variable.
    nd      : int
            Number of exogeneous signals.
    scalers : DataScalerConfig | ScalerType | str
            Configuration of the signal scaling strategies.
    SchedulingMap   : SchedulingMapConfig | dict
            Configuration of the scheduling map.
    StateEncoder    : StateEncoderConfig | dict | None
            Configuration of the state encoder. Default None (no encoder).

    """

    nx: int
    nu: int
    ny: int
    npar: int
    nreps: int | tuple[int, ...]
    nd: int
    scalers: DataScalerConfig | ScalerType | str = ScalerType.STANDARD
    SchedulingMap: SchedulingMapConfig | dict
    StateEncoder: StateEncoderConfig | dict | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError(f"nx must be > 0, got {self.nx}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.ny <= 0:
            raise ValueError(f"ny must be > 0, got {self.ny}")
        if self.npar <= 0:
            raise ValueError(f"npar must be > 0, got {self.npar}")
        if self.nd < 0:
            raise ValueError(f"nd must be >= 0, got {self.nd}")

        # Check if nreps is correct form
        _validate_nreps(self.npar, self.nreps)

        # Convert scalers to DataScalerConfig
        object.__setattr__(
            self, "scalers", DataScalerConfig.from_spec(self.scalers, nd=self.nd)
        )

        # Convert dict to SchedulingMapConfig if needed
        if isinstance(self.SchedulingMap, dict):
            object.__setattr__(
                self, "SchedulingMap", SchedulingMapConfig(**self.SchedulingMap)
            )

        # Convert dict to StateEncoderConfig if needed
        if isinstance(self.StateEncoder, dict):
            object.__setattr__(
                self, "StateEncoder", StateEncoderConfig(**self.StateEncoder)
            )


@dataclass(frozen=True, kw_only=True)
class RationalLPVLFRConfig:
    """Configuration for LPV models with rational-dependency linear fractional representation

    Parameters
    ----------
    nx      : int
            Number of states.
    nu      : int
            Number of inputs.
    ny      : int
            Number of outputs.
    npar    : int
            Number of scheduling variables.
    nreps   : int | tuple
            Number of repetitions of each scheduling variable.
    nd      : int
            Number of exogeneous signals.
    scalers : DataScalerConfig | ScalerType | str
            Configuration of the signal scaling strategies.
    SchedulingMap   : SchedulingMapConfig | dict
            Configuration of the scheduling map.
    StateEncoder    : StateEncoderConfig | dict | None
            Configuration of the state encoder. Default None (no encoder).

    """

    nx: int
    nu: int
    ny: int
    npar: int
    nreps: int | tuple
    nd: int
    scalers: DataScalerConfig | ScalerType | str = ScalerType.STANDARD
    SchedulingMap: SchedulingMapConfig | dict
    StateEncoder: StateEncoderConfig | dict | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError(f"nx must be > 0, got {self.nx}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.ny <= 0:
            raise ValueError(f"ny must be > 0, got {self.ny}")
        if self.npar <= 0:
            raise ValueError(f"npar must be > 0, got {self.npar}")
        if self.nd < 0:
            raise ValueError(f"nd must be >= 0, got {self.nd}")

        # Check if nreps is correct form
        _validate_nreps(self.npar, self.nreps)

        # Convert scalers to DataScalerConfig
        object.__setattr__(
            self, "scalers", DataScalerConfig.from_spec(self.scalers, nd=self.nd)
        )

        # Convert dict to SchedulingMapConfig if needed
        if isinstance(self.SchedulingMap, dict):
            object.__setattr__(
                self, "SchedulingMap", SchedulingMapConfig(**self.SchedulingMap)
            )

        # Convert dict to StateEncoderConfig if needed
        if isinstance(self.StateEncoder, dict):
            object.__setattr__(
                self, "StateEncoder", StateEncoderConfig(**self.StateEncoder)
            )


@dataclass(frozen=True, kw_only=True)
class NLLFRConfig:
    """Configuration for NL-LFR models

    Parameters
    ----------
    nx      : int
            Number of states.
    nu      : int
            Number of inputs.
    ny      : int
            Number of outputs.
    nw      : int
            Number of outputs of the DeltaMap.
    nz      : int
            Number of inputs of the DeltaMap.

    DeltaMap        : NetworkConfig | dict
                    Configuration of the delta map.
    scalers         : DataScalerConfig | ScalerType | str
                    Configuration of the signal scaling strategies.
    StateEncoder    : StateEncoderConfig | dict | None
                    Configuration of the state encoder. Default None (no encoder).

    """

    nx: int
    nu: int
    ny: int
    nw: int
    nz: int
    DeltaMap: NetworkConfig | dict
    scalers: DataScalerConfig | ScalerType | str = ScalerType.STANDARD
    StateEncoder: StateEncoderConfig | dict | None = None

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError(f"nx must be > 0, got {self.nx}")
        if self.nu <= 0:
            raise ValueError(f"nu must be > 0, got {self.nu}")
        if self.ny <= 0:
            raise ValueError(f"ny must be > 0, got {self.ny}")
        if self.nw <= 0:
            raise ValueError(f"nw must be > 0, got {self.nw}")
        if self.nz < 0:
            raise ValueError(f"nz must be >= 0, got {self.nz}")

        # Convert scalers to DataScalerConfig
        object.__setattr__(self, "scalers", DataScalerConfig.from_spec(self.scalers))

        # Convert dict to SchedulingMapConfig if needed
        if isinstance(self.DeltaMap, dict):
            object.__setattr__(self, "DeltaMap", NetworkConfig(**self.DeltaMap))

        # Convert dict to StateEncoderConfig if needed
        if isinstance(self.StateEncoder, dict):
            object.__setattr__(
                self, "StateEncoder", StateEncoderConfig(**self.StateEncoder)
            )


class AbstractModel(eqx.Module):
    """Base class for models simulated from scaled data.

    Subclasses provide scalers and the internal _simulate function.
    """

    scalers: eqx.AbstractVar[DataScaler]

    @abstractmethod
    def _simulate(self, data: iddata, x0: Array):
        raise NotImplementedError

    def simulate(self, data: iddata, x0: Array):
        """Simulate model given a dataset and initial state.
        Scales data according to u_scaler and y_scaler.

        Parameters
        ----------
        data    : iddata
            iddata containing input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Output signal generated by simulation.
        x_accum : Array, shape (N, nx)
            State trajectory generated by simulation.
        """
        data_scaled = self.scalers.transform(data)
        y_accum_scaled, x_accum = self._simulate(
            data_scaled, x0
        )  # state is unchanged under y and x scaling and shifting
        y_accum = self.scalers.y_scaler.inverse_transform(y_accum_scaled)
        return y_accum, x_accum


class AbstractEncoderModel(AbstractModel):
    """Base class for models with a state encoder for initial state estimation.

    Subclasses provide a StateEncoder, which is None if no encoder is used.
    """

    StateEncoder: eqx.AbstractVar[NNStateEncoder | None]

    @property
    def n_hist(self) -> int | None:
        """Number of past samples used by the state encoder."""
        if self.StateEncoder is None:
            return None
        return self.StateEncoder.n_hist

    def encoder_state_prediction(self, data: iddata):
        """Estimate the state trajectory using the state encoder.
        Scales data according to u_scaler and y_scaler.

        The state at each time step is estimated from the n_hist preceding
        inputs and outputs, so the first n_hist time steps are not estimated.

        Parameters
        ----------
        data    : iddata
            iddata containing input and output data.

        Returns
        -------
        x_accum : Array, shape (N - n_hist, nx)
            Estimated states at time steps n_hist, ..., N - 1.
        """
        data_scaled = self.scalers.transform(data)
        data_split, _ = create_encoder_iddata(
            data_scaled, T=1, overlap=0, n_hist=self.n_hist
        )
        x_accum = self._encoder_state_prediction(data_split)
        return x_accum

    def _encoder_state_prediction(self, data: iddata):
        """Internal state trajectory estimation, expects scaled data with history windows.

        Parameters
        ----------
        data    : iddata
            iddata with M stacked history windows, e.g. U_hist of shape
            (M, n_hist, nu) and Y_hist of shape (M, n_hist, ny).

        Returns
        -------
        x_accum : Array, shape (M, nx)
            Estimated state following each history window.
        """
        x_accum = jax.vmap(self._encoder_initial_state_prediction)(data)
        return x_accum

    def encoder_initial_state_prediction(self, data: iddata, num_past: int = 0):
        """Estimate the initial state using the state encoder.
        Scales data according to u_scaler and y_scaler.

        The first n_hist samples of data are used by the state encoder, the
        estimated state corresponds to time step n_hist.

        Parameters
        ----------
        data        : iddata
            iddata containing input and output data, with N > n_hist.
        num_past    : int, optional
            Currently unused.

        Returns
        -------
        x0  : Array, shape (nx,)
            Estimated state at time step n_hist.
        """
        data_scaled = self.scalers.transform(data)
        data_with_hist, _ = create_encoder_iddata(
            data_scaled,
            T=(data.U.shape[0] - self.n_hist),
            overlap=0,
            n_hist=self.n_hist,
        )
        x0 = self._encoder_initial_state_prediction(data_with_hist)
        return x0

    def _encoder_initial_state_prediction(self, data: iddata):
        """Internal initial state estimation, expects scaled data with a history window.

        Parameters
        ----------
        data    : iddata
            iddata with history windows Y_hist and U_hist (and D_hist if
            exogenous signals are used) of length n_hist.

        Returns
        -------
        x0  : Array, shape (nx,)
            Estimated state following the history window.
        """
        return self.StateEncoder(data)


class ModelLinStateSpace(AbstractModel):
    """Linear state-space model class

    Parameters
    ----------
    A       : jnp.ndarray, shape (nx, nx)
            State transition matrix.
    B_u     : jnp.ndarray, shape (nx, nu)
            Input-to-state matrix.
    C_y     : jnp.ndarray, shape (ny, nx)
            State-to-output matrix.
    D_yu    : jnp.ndarray, shape (ny, nu)
            Input-tooutput matrix.
    scalers : DataScaler
            Instance of DataScaler class specifying scaling type per signal.
    config  : LinStateSpaceConfig
            Configuration used to create this model.

    """

    A: jnp.ndarray
    B_u: jnp.ndarray
    C_y: jnp.ndarray
    D_yu: jnp.ndarray

    scalers: DataScaler
    config: LinStateSpaceConfig = eqx.field(static=True)

    def __init__(
        self,
        config: LinStateSpaceConfig,
        key: PRNGKeyArray,
        *,
        scalers: DataScaler = None,
    ):
        self.config = config
        A, B_u, C_y, D_yu = init_lti_params(config.nx, config.nu, config.ny, key)
        self.A = A
        self.B_u = B_u
        self.C_y = C_y
        self.D_yu = D_yu

        self.scalers = DataScaler(config.scalers)
        if scalers is not None:
            self.scalers = scalers

    @classmethod
    def create(
        cls,
        nx: int,
        nu: int,
        ny: int,
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
    ):
        """Create an instance of ModelLinStateSpace

        Parameters
        ----------
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        scaler_config : ScalerType
            Type of data scaler, by default "standard".

        Returns
        -------
        ModelLinStateSpace
            Linear state-space model
        """
        config = LinStateSpaceConfig(nx=nx, nu=nu, ny=ny, scalers=scaler_config)
        return cls(config, jax.random.PRNGKey(KEY))

    def _simulate(self, data: iddata, x0: Array):
        """Internal simulation function, expects scaled data.

        Parameters
        ----------
        data    : iddata
            iddata containing scaled input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Scaled output signal.
        x_accum : Array, shape (N, nx)
            State trajectory.
        """

        def _make_step(state, input):
            x = state
            u = input
            y = self.C_y @ x + self.D_yu @ u
            x_next = self.A @ x + self.B_u @ u
            x_next = jnp.clip(x_next, -1000, 1000)
            return x_next, (y, x)

        x0 = x0.squeeze()  # ensures x0 is (nx,) for scan
        _, (y_accum, x_accum) = jax.lax.scan(_make_step, x0, data.U)
        return y_accum, x_accum


class ModelAffineLPVLFR(AbstractEncoderModel):
    """Model class for LPV models with affine-dependency linear fractional representation

    Parameters
    ----------
    A       : jnp.ndarray, shape (nx, nx)
            State transition matrix.
    B_u     : jnp.ndarray, shape (nx, nu)
            Input-to-state matrix.
    C_y     : jnp.ndarray, shape (ny, nx)
            State-to-output matrix.
    D_yu    : jnp.ndarray, shape (ny, nu)
            Input-to-output matrix.
    B_w     : jnp.ndarray, shape (nx, nw)
            Latent-to-state matrix.
    C_z     : jnp.ndarray, shape (nw, nx)
            State-to-latent matrix.
    D_zu    : jnp.ndarray, shape (nw, nu)
            Input-to-latent matrix.
    D_yw    : jnp.ndarray, shape (ny, nw)
            Latent-to-output matrix.
    nreps           : int | tuple
                    Number of repetitions of the delta block.
    SchedulingMap   : AbstractSchedulingMap
                    Scheduling map used with LPV-LFR model.
    StateEncoder    : NNStateEncoder
                    State encoder used for initialial state estimation in encoder-based multiple-shooting.
    scalers         : DataScaler
                    Instance of DataScaler class specifying scaling type per signal.
    config          : AffineLPVLFRConfig
                    Configuration used to create this model.


    """

    A: jnp.ndarray
    B_u: jnp.ndarray
    C_y: jnp.ndarray
    D_yu: jnp.ndarray
    B_w: jnp.ndarray
    C_z: jnp.ndarray
    D_zu: jnp.ndarray
    D_yw: jnp.ndarray

    nreps: int = eqx.field(static=True)

    SchedulingMap: AbstractSchedulingMap
    StateEncoder: NNStateEncoder | None = None

    scalers: DataScaler
    config: AffineLPVLFRConfig = eqx.field(static=True)

    def __init__(
        self,
        config: AffineLPVLFRConfig,
        key: PRNGKeyArray,
        *,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
        scalers: DataScaler = None,
    ):
        self.config = config
        if isinstance(config.nreps, int):
            nw = config.npar * config.nreps
        elif isinstance(config.nreps, tuple):
            nw = sum(config.nreps)
        else:
            raise TypeError("nreps must be int or tuple")

        lfr_key, sched_key, enc_key = jax.random.split(key, 3)
        A, B_w, B_u, C_z, D_zu, C_y, D_yw, D_yu = init_lfr_params(
            config.nx, config.nu, config.ny, nw, lfr_key
        )

        if init_sys is not None:
            A, B_u, C_y, D_yu = init_sys

        self.nreps = config.nreps
        self.A = A
        self.B_w = B_w
        self.B_u = B_u
        self.C_z = C_z
        self.D_zu = D_zu
        self.C_y = C_y
        self.D_yw = D_yw
        self.D_yu = D_yu

        self.scalers = DataScaler(config.scalers)
        if scalers is not None:
            self.scalers = scalers

        self.SchedulingMap = NNSchedulingMap(
            config.SchedulingMap,
            config.nx + config.nu + config.nd,
            config.npar,
            sched_key,
        )

        if config.StateEncoder is not None:
            self.StateEncoder = NNStateEncoder(
                config.StateEncoder,
                config.nx,
                config.nu,
                config.ny,
                config.nd,
                enc_key,
            )

    @classmethod
    def create(
        cls,
        nx: int,
        nu: int,
        ny: int,
        nd: int,
        npar: int,
        nreps: int | tuple[int, ...],
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        scheduling_config: SchedulingMapConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of ModelAffineLPVLFR

        Parameters
        ----------
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nd : int
            Number of exogeneous signals.
        npar : int
            Number of scheduling variables.
        nreps : int | tuple[int, ...]
            Number of repetitions of each scheduling variable.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        scheduling_config : SchedulingMapConfig, optional
            Configuration of the scheduling map.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize the LPV-LFR model.

        Returns
        -------
        ModelAffineLPVLFR
            Model class for LPV models with affine-dependency linear fractional representation
        """
        if scheduling_config is None:
            scheduling_config = SchedulingMapConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)
        aff_config = AffineLPVLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            npar=npar,
            nreps=nreps,
            nd=nd,
            scalers=scaler_config,
            SchedulingMap=scheduling_config,
            StateEncoder=encoder_config,
        )
        return cls(aff_config, jax.random.PRNGKey(KEY), init_sys=init_sys)

    def _simulate(self, data: iddata, x0: Array):
        """Internal simulation function, expects scaled data.

        Parameters
        ----------
        data    : iddata
            iddata containing scaled input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Scaled output signal.
        x_accum : Array, shape (N, nx)
            State trajectory.
        """
        A = self.A
        B_u = self.B_u
        B_w = self.B_w
        C_y = self.C_y
        D_yu = self.D_yu
        D_yw = self.D_yw
        C_z = self.C_z
        D_zu = self.D_zu
        SchedulingMap = self.SchedulingMap
        nreps = self.nreps

        def _make_step(state, input):
            x = state
            u, d = input

            p = SchedulingMap._evaluate(x, u, d)
            delta = jnp.repeat(p, nreps)
            z = C_z @ x + D_zu @ u
            w = delta * z
            y = C_y @ x + D_yu @ u + D_yw @ w
            x_next = A @ x + B_u @ u + B_w @ w
            x_next = jnp.clip(x_next, -1000, 1000)

            return x_next, (y, x)

        x0 = x0.squeeze()  # ensures x0 is (nx,) for scan
        if self.config.nd > 0:
            d = data.D
        else:
            d = jnp.zeros((data.U.shape[0], 0))

        _, (y_accum, x_accum) = jax.lax.scan(_make_step, x0, (data.U, d))
        return y_accum, x_accum


class ModelRationalLPVLFR(AbstractEncoderModel):
    """Model class for LPV models with rational-dependency linear fractional representation.
    Uses direct-parameterization which guarantees well-posedness as defined in https://arxiv.org/abs/2605.12203.

    Parameters
    ----------
    A       : jnp.ndarray, shape (nx, nx)
            State transition matrix.
    B_u     : jnp.ndarray, shape (nx, nu)
            Input-to-state matrix.
    C_y     : jnp.ndarray, shape (ny, nx)
            State-to-output matrix.
    D_yu    : jnp.ndarray, shape (ny, nu)
            Input-to-output matrix.
    B_w     : jnp.ndarray, shape (nx, nw)
            Latent-to-state matrix.
    C_z     : jnp.ndarray, shape (nw, nx)
            State-to-latent matrix.
    D_zu    : jnp.ndarray, shape (nw, nu)
            Input-to-latent matrix.
    D_yw    : jnp.ndarray, shape (ny, nw)
            Latent-to-output matrix.
    D_zw_A  : jnp.ndarray, shape (nw, nw)
            Positive-definite parameters of D_zw matrix.
    D_zw_B  : jnp.ndarray, shape (nw, nw)
            Semi-definite parameters of D_zw matrix.
    D_zw_d  : jnp.ndarray, shape (nw,)
            Scaling parameters of D_zw matrix.
    nreps           : int | tuple
                    Number of repetitions of the delta block.
    SchedulingMap   : AbstractSchedulingMap
                    Scheduling map used with LPV-LFR model.
    StateEncoder    : NNStateEncoder
                    State encoder used for initialial state estimation in encoder-based multiple-shooting.
    scalers         : DataScaler
                    Instance of DataScaler class specifying scaling type per signal.
    config          : RationalLPVLFRConfig
                    Configuration used to create this model.

    """

    A: jnp.ndarray
    B_u: jnp.ndarray
    C_y: jnp.ndarray
    D_yu: jnp.ndarray
    B_w: jnp.ndarray
    C_z: jnp.ndarray
    D_zu: jnp.ndarray
    D_yw: jnp.ndarray
    D_zw_A: jnp.ndarray
    D_zw_B: jnp.ndarray
    D_zw_d: jnp.ndarray

    nreps: int = eqx.field(static=True)
    nw: int = eqx.field(static=True)

    SchedulingMap: AbstractSchedulingMap
    StateEncoder: NNStateEncoder | None = None

    scalers: DataScaler
    config: RationalLPVLFRConfig = eqx.field(static=True)

    def __init__(
        self,
        config,
        key: PRNGKeyArray,
        *,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
        scalers: DataScaler = None,
    ):
        self.config = config
        if isinstance(config.nreps, int):
            nw = config.npar * config.nreps
        elif isinstance(config.nreps, tuple):
            nw = sum(config.nreps)
        else:
            raise TypeError("nreps must be int or tuple")

        lfr_key, sched_key, enc_key = jax.random.split(key, 3)
        A, B_w, B_u, C_z, D_zu, C_y, D_yw, D_yu = init_lfr_params(
            config.nx, config.nu, config.ny, nw, lfr_key
        )
        if init_sys is not None:
            A, B_u, C_y, D_yu = init_sys

        D_zw_A_mat = jnp.sqrt(-jnp.log(0.5)) * jnp.eye(nw)
        D_zw_A = D_zw_A_mat[jnp.tril_indices(nw)]

        D_zw_B_mat = jnp.zeros((nw, nw))
        D_zw_B = D_zw_B_mat[jnp.triu_indices(nw, 1)]

        D_zw_d = jnp.zeros((nw, 1))

        self.nreps = config.nreps
        self.nw = nw
        self.A = A
        self.B_w = B_w
        self.B_u = B_u
        self.C_z = C_z
        self.D_zu = D_zu
        self.C_y = C_y
        self.D_yw = D_yw
        self.D_yu = D_yu
        self.D_zw_A = D_zw_A
        self.D_zw_B = D_zw_B
        self.D_zw_d = D_zw_d

        self.scalers = DataScaler(config.scalers)
        if scalers is not None:
            self.scalers = scalers

        self.SchedulingMap = NNSchedulingMap(
            config.SchedulingMap,
            config.nx + config.nu + config.nd,
            config.npar,
            sched_key,
        )

        if config.StateEncoder is not None:
            self.StateEncoder = NNStateEncoder(
                config.StateEncoder,
                config.nx,
                config.nu,
                config.ny,
                config.nd,
                enc_key,
            )

    @classmethod
    def create(
        cls,
        nx: int,
        nu: int,
        ny: int,
        nd: int,
        npar: int,
        nreps: int | tuple[int, ...],
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        scheduling_config: SchedulingMapConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of ModelRationalLPVLFR

        Parameters
        ----------
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nd : int
            Number of exogeneous signals.
        npar : int
            Number of scheduling variables.
        nreps : int | tuple[int, ...]
            Number of repetitions of each scheduling variable.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        scheduling_config : SchedulingMapConfig, optional
            Configuration of the scheduling map.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize the LPV-LFR model.

        Returns
        -------
        ModelRationalLPVLFR
            Model class for LPV models with rational-dependency linear fractional representation
        """
        if scheduling_config is None:
            scheduling_config = SchedulingMapConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)
        rat_config = RationalLPVLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            npar=npar,
            nreps=nreps,
            nd=nd,
            scalers=scaler_config,
            SchedulingMap=scheduling_config,
            StateEncoder=encoder_config,
        )
        return cls(rat_config, jax.random.PRNGKey(KEY), init_sys=init_sys)

    def _simulate(self, data: iddata, x0: Array):
        """Internal simulation function, expects scaled data.

        Parameters
        ----------
        data    : iddata
            iddata containing scaled input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Scaled output signal.
        x_accum : Array, shape (N, nx)
            State trajectory.
        """
        A = self.A
        B_u = self.B_u
        B_w = self.B_w
        C_y = self.C_y
        D_yu = self.D_yu
        D_yw = self.D_yw
        C_z = self.C_z
        D_zu = self.D_zu
        D_zw_A = self.D_zw_A
        D_zw_B = self.D_zw_B
        D_zw_d = self.D_zw_d

        nreps = self.nreps
        nw = self.nw

        D_zw_A_mat = jnp.zeros((nw, nw)).at[jnp.tril_indices(nw)].set(D_zw_A)
        D_zw_B_mat = jnp.zeros((nw, nw)).at[jnp.triu_indices(nw, 1)].set(D_zw_B)

        D_zw_Psi = jnp.exp(D_zw_d)
        I_w = jnp.eye(nw)
        D_zw = jax.scipy.linalg.expm(
            -D_zw_Psi
            * (D_zw_A_mat @ D_zw_A_mat.T + D_zw_B_mat - D_zw_B_mat.T + 1e-5 * I_w)
        )

        SchedulingMap = self.SchedulingMap

        def _make_step(state, input):
            x = state
            u, d = input

            p = SchedulingMap._evaluate(x, u, d)
            delta = jnp.repeat(p, nreps)
            M = I_w - D_zw * delta
            rhs = C_z @ x + D_zu @ u
            if nw == 3:
                z = _solve_3x3_cramer(M, rhs)
            else:
                z = jnp.linalg.solve(M, rhs)
            w = delta * z
            y = C_y @ x + D_yu @ u + D_yw @ w
            x_next = A @ x + B_u @ u + B_w @ w
            x_next = jnp.clip(x_next, -1000, 1000)

            return x_next, (y, x)

        x0 = x0.squeeze()  # ensures x0 is (nx,) for scan
        if self.config.nd > 0:
            d = data.D
        else:
            d = jnp.zeros((data.U.shape[0], 0))

        _, (y_accum, x_accum) = jax.lax.scan(_make_step, x0, (data.U, d))
        return y_accum, x_accum


class ModelNLLFR(AbstractEncoderModel):
    """Model class for NL-LFR models, well-posedness is guaranteed by enforcing D_zw = 0.

    Parameters
    ----------
    A       : jnp.ndarray, shape (nx, nx)
            State transition matrix.
    B_u     : jnp.ndarray, shape (nx, nu)
            Input-to-state matrix.
    C_y     : jnp.ndarray, shape (ny, nx)
            State-to-output matrix.
    D_yu    : jnp.ndarray, shape (ny, nu)
            Input-to-output matrix.
    B_w     : jnp.ndarray, shape (nx, nw)
            Latent-to-state matrix.
    C_z     : jnp.ndarray, shape (nz, nx)
            State-to-latent matrix.
    D_zu    : jnp.ndarray, shape (nz, nu)
            Input-to-latent matrix.
    D_yw    : jnp.ndarray, shape (ny, nw)
            Latent-to-output matrix.

    DeltaMap        : eqx.Module
                    Delta map ``w = Delta(z)`` used to represent nonlinearities.
    StateEncoder    : NNStateEncoder
                    State encoder used for initial state estimation in encoder-based multiple-shooting.
    scalers         : DataScaler
                    Instance of DataScaler class specifying scaling type per signal.
    config          : NLLFRConfig
                    Configuration used to create this model.


    """

    A: jnp.ndarray
    B_u: jnp.ndarray
    C_y: jnp.ndarray
    D_yu: jnp.ndarray
    B_w: jnp.ndarray
    C_z: jnp.ndarray
    D_zu: jnp.ndarray
    D_yw: jnp.ndarray

    DeltaMap: eqx.Module

    StateEncoder: NNStateEncoder | None = None

    scalers: DataScaler
    config: NLLFRConfig = eqx.field(static=True)

    def __init__(
        self,
        config: NLLFRConfig,
        key: PRNGKeyArray,
        *,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
        scalers: DataScaler = None,
    ):
        self.config = config

        lfr_key, delta_key, enc_key = jax.random.split(key, 3)
        A, B_w, B_u, C_z, D_zu, C_y, D_yw, D_yu = init_lfr_params(
            config.nx, config.nu, config.ny, config.nw, lfr_key, nz=config.nz
        )

        if init_sys is not None:
            A, B_u, C_y, D_yu = init_sys

        self.A = A
        self.B_w = B_w
        self.B_u = B_u
        self.C_z = C_z
        self.D_zu = D_zu
        self.C_y = C_y
        self.D_yw = D_yw
        self.D_yu = D_yu

        self.scalers = DataScaler(config.scalers)
        if scalers is not None:
            self.scalers = scalers

        self.DeltaMap = NETWORKS[config.DeltaMap.network](
            in_size=config.nz,
            out_size=config.nw,
            width_size=config.DeltaMap.neurons_per_layer,
            depth=config.DeltaMap.layers,
            activation=ACTIVATIONS[config.DeltaMap.activation],
            final_activation=jax.nn.tanh if config.DeltaMap.clip else (lambda x: x),
            key=delta_key,
        )

        if config.StateEncoder is not None:
            self.StateEncoder = NNStateEncoder(
                config.StateEncoder, config.nx, config.nu, config.ny, 0, enc_key
            )

    @classmethod
    def create(
        cls,
        nx: int,
        nu: int,
        ny: int,
        nw: int,
        nz: int,
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        deltamap_config: NetworkConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of ModelNLLFR

        Parameters
        ----------
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nw : int
            Number of outputs of the DeltaMap.
        nz : int
            Number of inputs of the DeltaMap.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        deltamap_config : NetworkConfig, optional
            Configuration of the DeltaMap.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize the NLLFR model.

        Returns
        -------
        ModelNLLFR
            Model class for NL models with Linear Fractional Representation.
        """
        if deltamap_config is None:
            deltamap_config = NetworkConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)
        nllfr_config = NLLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            nw=nw,
            nz=nz,
            scalers=scaler_config,
            DeltaMap=deltamap_config,
            StateEncoder=encoder_config,
        )
        return cls(nllfr_config, jax.random.PRNGKey(KEY), init_sys=init_sys)

    def _simulate(self, data: iddata, x0: Array):
        """Internal simulation function, expects scaled data.

        Parameters
        ----------
        data    : iddata
            iddata containing scaled input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Scaled output signal.
        x_accum : Array, shape (N, nx)
            State trajectory.
        """
        A = self.A
        B_u = self.B_u
        B_w = self.B_w
        C_y = self.C_y
        D_yu = self.D_yu
        D_yw = self.D_yw
        C_z = self.C_z
        D_zu = self.D_zu
        DeltaMap = self.DeltaMap

        def _make_step(state, input):
            x = state
            u = input

            z = C_z @ x + D_zu @ u
            w = DeltaMap(z)
            y = C_y @ x + D_yu @ u + D_yw @ w
            x_next = A @ x + B_u @ u + B_w @ w
            x_next = jnp.clip(x_next, -1000, 1000)

            return x_next, (y, x)

        x0 = x0.squeeze()
        _, (y_accum, x_accum) = jax.lax.scan(_make_step, x0, data.U)
        return y_accum, x_accum

    def rescale(self, data: iddata | iddata_collection | Sequence[iddata]):
        """Rescale the model such that the DeltaMap inputs z have unit standard deviation.
        Initializes the scalers using data if they are not yet initialized.

        The model is simulated from a zero initial state, after which the rows
        of C_z and D_zu are divided by the standard deviation of the
        corresponding entries of z.

        Parameters
        ----------
        data    : iddata | iddata_collection | Sequence[iddata]
            iddata containing input and output data, or multiple experiments
            whose z trajectories are pooled.

        Returns
        -------
        ModelNLLFR
            Rescaled model.

        Raises
        ------
        ValueError
            If any entry of z has zero standard deviation.
        """
        if not isinstance(data, iddata):
            data = as_collection(data)
        model = initialize_scalers(self, data)
        data_scaled = model.scalers.transform(data)

        def latent_fn(d):
            _, x = model._simulate(d, jnp.zeros((model.config.nx,)))
            return x @ model.C_z.T + d.U @ model.D_zu.T

        # For multiple experiments, z is concatenated along time
        if isinstance(data_scaled, iddata):
            z = latent_fn(data_scaled)
        else:
            z = jnp.concatenate([latent_fn(d) for d in data_scaled], axis=0)
        C_z_new, D_zu_new = scale_nllfr(model.C_z, model.D_zu, z)
        return eqx.tree_at(
            lambda m: (m.C_z, m.D_zu),
            model,
            (C_z_new, D_zu_new),
        )


class AbstractEnsemble(eqx.Module):
    """Base class for ensembles of models, subclasses set _model_cls.

    Parameters
    ----------
    models  :   AbstractModel
                vmapped version of _model_cls
    scalers :   DataScaler (property)
                Access to the DataScaler instance specifying scaling type per signal.
                All models in the ensemble share the same scalers.
    config  :   LinStateSpaceConfig | AffineLPVLFRConfig | RationalLPVLFRConfig | NLLFRConfig
                Configuration used to create this ensemble.
    n_models:   int
                Number of models in the ensemble.

    """

    models: AbstractModel
    config: (
        LinStateSpaceConfig | AffineLPVLFRConfig | RationalLPVLFRConfig | NLLFRConfig
    ) = eqx.field(static=True)
    n_models: int = eqx.field(static=True)

    _model_cls: ClassVar[type[AbstractModel]]

    def __init__(
        self,
        n_models: int,
        config,
        key: PRNGKeyArray,
        *,
        scalers: DataScaler = None,
        **model_kwargs,
    ):
        """
        Parameters
        ----------
        n_models        : int
            Number of models in the ensemble.
        config          : LinStateSpaceConfig | AffineLPVLFRConfig | RationalLPVLFRConfig | NLLFRConfig
            Configuration of the models, matching _model_cls.
        key             : PRNGKeyArray
            Key used for random initialization, split over the models.
        scalers         : DataScaler, optional
            Scalers shared by all models.
        **model_kwargs
            Passed to _model_cls, e.g. init_sys for LFR models.
        """
        self.n_models = n_models
        self.config = config

        if scalers is None:
            scalers = DataScaler(config.scalers)

        def model_fn(key):
            return self._model_cls(config, key, scalers=scalers, **model_kwargs)

        keys = jax.random.split(key, n_models)
        self.models = eqx.filter_vmap(model_fn)(keys)

    @property
    def scalers(self) -> DataScaler:
        """Access scalers via first model (all ensemble models share same scalers)."""
        return jax.tree_util.tree_map(lambda x: x[0], self.models.scalers)

    def with_scalers(self, scalers: DataScaler):
        """Return a copy of the ensemble in which all models use scalers.

        Parameters
        ----------
        scalers : DataScaler
            Unbatched scalers, broadcast over the models.

        Returns
        -------
        AbstractEnsemble
            Ensemble with updated scalers.
        """
        scalers_batched = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.n_models,) + jnp.shape(x)), scalers
        )
        return eqx.tree_at(lambda e: e.models.scalers, self, scalers_batched)

    def simulate(self, data: iddata, x0: Array):
        """Simulate all models in the ensemble given a dataset and initial state.
        Scales data according to u_scaler and y_scaler.

        Parameters
        ----------
        data    : iddata
            iddata containing input and output data.
        x0      : Array, shape (nx,) or (n_models, nx)
            Initial state, shared by all models, or one initial state per
            model (e.g. as learned by optimize_ensemble).

        Returns
        -------
        y_accum : Array, shape (n_models, N, ny)
            Output signals generated by simulation.
        x_accum : Array, shape (n_models, N, nx)
            State trajectories generated by simulation.
        """
        if x0.ndim == 2 and x0.shape[0] == self.n_models:
            return eqx.filter_vmap(lambda m, x: m.simulate(data, x))(self.models, x0)
        return eqx.filter_vmap(lambda m: m.simulate(data, x0))(self.models)

    def _simulate(self, data: iddata, x0: Array):
        """Internal simulation function, expects scaled data.

        Simulates a single model, so it should be called inside a vmap over
        the ensemble (as done by optimize_ensemble).

        Parameters
        ----------
        data    : iddata
            iddata containing scaled input and output data.
        x0      : Array, shape (nx,)
            Initial state.

        Returns
        -------
        y_accum : Array, shape (N, ny)
            Scaled output signal.
        x_accum : Array, shape (N, nx)
            State trajectory.
        """
        return self.models._simulate(data, x0)


class EnsembleLinStateSpace(AbstractEnsemble):
    """Ensemble of ModelLinStateSpace, see AbstractEnsemble."""

    _model_cls = ModelLinStateSpace

    @classmethod
    def create(
        cls,
        n_models: int,
        nx: int,
        nu: int,
        ny: int,
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
    ):
        """Create an instance of EnsembleLinStateSpace

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".

        Returns
        -------
        EnsembleLinStateSpace
            Ensemble of linear state-space models.
        """
        config = LinStateSpaceConfig(nx=nx, nu=nu, ny=ny, scalers=scaler_config)
        return cls(n_models, config, jax.random.PRNGKey(KEY))


class EnsembleAffineLPVLFR(AbstractEnsemble):
    """Ensemble of ModelAffineLPVLFR, see AbstractEnsemble."""

    _model_cls = ModelAffineLPVLFR

    @classmethod
    def create(
        cls,
        n_models: int,
        nx: int,
        nu: int,
        ny: int,
        nd: int,
        npar: int,
        nreps: int | tuple[int, ...],
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        scheduling_config: SchedulingMapConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of EnsembleAffineLPVLFR

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nd : int
            Number of exogeneous signals.
        npar : int
            Number of scheduling variables.
        nreps : int | tuple[int, ...]
            Number of repetitions of each scheduling variable.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        scheduling_config : SchedulingMapConfig, optional
            Configuration of the scheduling map.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize all LPV-LFR models.

        Returns
        -------
        EnsembleAffineLPVLFR
            Ensemble of LPV models with affine-dependency linear fractional representation
        """
        if scheduling_config is None:
            scheduling_config = SchedulingMapConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)

        aff_config = AffineLPVLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            npar=npar,
            nreps=nreps,
            nd=nd,
            scalers=scaler_config,
            SchedulingMap=scheduling_config,
            StateEncoder=encoder_config,
        )
        return cls(n_models, aff_config, jax.random.PRNGKey(KEY), init_sys=init_sys)


class EnsembleRationalLPVLFR(AbstractEnsemble):
    """Ensemble of ModelRationalLPVLFR, see AbstractEnsemble."""

    _model_cls = ModelRationalLPVLFR

    @classmethod
    def create(
        cls,
        n_models: int,
        nx: int,
        nu: int,
        ny: int,
        nd: int,
        npar: int,
        nreps: int | tuple[int, ...],
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        scheduling_config: SchedulingMapConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of EnsembleRationalLPVLFR

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nd : int
            Number of exogeneous signals.
        npar : int
            Number of scheduling variables.
        nreps : int | tuple[int, ...]
            Number of repetitions of each scheduling variable.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        scheduling_config : SchedulingMapConfig, optional
            Configuration of the scheduling map.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize all LPV-LFR models.

        Returns
        -------
        EnsembleRationalLPVLFR
            Ensemble of LPV models with rational-dependency linear fractional representation
        """
        if scheduling_config is None:
            scheduling_config = SchedulingMapConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)

        rat_config = RationalLPVLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            npar=npar,
            nreps=nreps,
            nd=nd,
            scalers=scaler_config,
            SchedulingMap=scheduling_config,
            StateEncoder=encoder_config,
        )
        return cls(n_models, rat_config, jax.random.PRNGKey(KEY), init_sys=init_sys)


class EnsembleNLLFR(AbstractEnsemble):
    """Ensemble of ModelNLLFR, see AbstractEnsemble."""

    _model_cls = ModelNLLFR

    @classmethod
    def create(
        cls,
        n_models: int,
        nx: int,
        nu: int,
        ny: int,
        nw: int,
        nz: int,
        *,
        scaler_config: ScalerType = ScalerType.STANDARD,
        deltamap_config: NetworkConfig | None = None,
        encoder_config: StateEncoderConfig | None = None,
        init_sys: tuple[Array, Array, Array, Array] | None = None,
    ):
        """Create an instance of EnsembleNLLFR

        Parameters
        ----------
        n_models : int
            Number of models in the ensemble.
        nx : int
            Number of states.
        nu : int
            Number of inputs.
        ny : int
            Number of outputs.
        nw : int
            Number of outputs of the DeltaMap.
        nz : int
            Number of inputs of the DeltaMap.
        scaler_config : ScalerType, optional
            Type of data scaler, by default "standard".
        deltamap_config : NetworkConfig, optional
            Configuration of the DeltaMap.
        encoder_config : StateEncoderConfig, optional
            Configuration of the state encoder.
        init_sys : tuple[Array, Array, Array, Array], optional
            A, Bu, Cy, and Dyu matrices used to initialize all NLLFR models.

        Returns
        -------
        EnsembleNLLFR
            Ensemble of NL models with Linear Fractional Representation.
        """
        if deltamap_config is None:
            deltamap_config = NetworkConfig()
        if encoder_config is None:
            encoder_config = StateEncoderConfig(n_hist=nx * 4 + 1)

        nllfr_config = NLLFRConfig(
            nx=nx,
            nu=nu,
            ny=ny,
            nw=nw,
            nz=nz,
            scalers=scaler_config,
            DeltaMap=deltamap_config,
            StateEncoder=encoder_config,
        )
        return cls(n_models, nllfr_config, jax.random.PRNGKey(KEY), init_sys=init_sys)

    def rescale(self, data: iddata | iddata_collection | Sequence[iddata]):
        """Rescale all models such that the DeltaMap inputs z have unit standard deviation.
        Initializes the shared scalers using data if they are not yet initialized.

        Each model is rescaled as in ModelNLLFR.rescale.

        Parameters
        ----------
        data    : iddata | iddata_collection | Sequence[iddata]
            iddata containing input and output data, or multiple experiments
            whose z trajectories are pooled.

        Returns
        -------
        EnsembleNLLFR
            Rescaled ensemble.

        Raises
        ------
        ValueError
            If any entry of z has zero standard deviation for one of the models.
        """
        if not isinstance(data, iddata):
            data = as_collection(data)
        ensemble = self
        if not ensemble.scalers.is_initialized:
            ensemble = ensemble.with_scalers(ensemble.scalers.initialize(data))

        data_scaled = ensemble.scalers.transform(data)

        def latent_fn(m, d):
            _, x = m._simulate(d, jnp.zeros((ensemble.config.nx,)))
            return x @ m.C_z.T + d.U @ m.D_zu.T

        # For multiple experiments, z is concatenated along time
        experiments = (
            [data_scaled] if isinstance(data_scaled, iddata) else list(data_scaled)
        )
        z = jnp.concatenate(
            [
                eqx.filter_vmap(lambda m: latent_fn(m, d))(ensemble.models)
                for d in experiments
            ],
            axis=1,
        )

        # scale_nllfr is not traceable, so scale each model separately
        C_z_new, D_zu_new = zip(
            *(
                scale_nllfr(ensemble.models.C_z[i], ensemble.models.D_zu[i], z[i])
                for i in range(ensemble.n_models)
            )
        )
        return eqx.tree_at(
            lambda e: (e.models.C_z, e.models.D_zu),
            ensemble,
            (jnp.stack(C_z_new), jnp.stack(D_zu_new)),
        )
