"""Identax: efficient learning of linear, parameter-varying, and nonlinear systems in JAX.

Identax identifies discrete-time models from input-output data by
prediction error minimization: LTI state-space (LTI-SS) models, LPV models
in linear fractional representation (LPV-LFR, affine or rational scheduling
dependency) and nonlinear LFR (NL-LFR) models. Models are Equinox modules
and are trained with ``optimize`` (one model) or ``optimize_ensemble``
(many models in parallel), using single shooting or multiple shooting
(UMS or EBMS).

Note: importing identax enables 64-bit precision in JAX globally
(``jax_enable_x64``), which affects all JAX code in the same process.

Example
-------
>>> import identax as idx
>>> config = idx.LinStateSpaceConfig(nx=2, nu=1, ny=1)
>>> model = idx.ModelLinStateSpace(config, jax.random.PRNGKey(0))
>>> model, x0, history = idx.optimize(model, train_data, x0, idx.OptimConfig())
>>> y_hat, _ = model.simulate(test_data, x0_test)
"""

from importlib import metadata as _metadata

import jax

# Enforce JAX x64 precision at import time, before any submodule creates arrays
jax.config.update("jax_enable_x64", True)

from .models import (
    LinStateSpaceConfig,
    AffineLPVLFRConfig,
    RationalLPVLFRConfig,
    NLLFRConfig,
    ModelLinStateSpace,
    ModelAffineLPVLFR,
    ModelRationalLPVLFR,
    ModelNLLFR,
    EnsembleLinStateSpace,
    EnsembleAffineLPVLFR,
    EnsembleRationalLPVLFR,
    EnsembleNLLFR,
)

from .optim import (
    OptimConfig,
    OptimHistory,
    optimize,
    optimize_ensemble,
    learn_initial_state,
)

from .dataprocessing import (
    iddata,
    iddata_collection,
    simulate_collection,
    DataScalerConfig,
)

from .networks import (
    SchedulingMapConfig,
    StateEncoderConfig,
    NetworkConfig,
)

from .serialization import save_model, load_model

from ._model_init import scale_lti_ss

from . import metrics

try:
    __version__ = _metadata.version("identax")
except _metadata.PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "LinStateSpaceConfig",
    "AffineLPVLFRConfig",
    "RationalLPVLFRConfig",
    "NLLFRConfig",
    "ModelLinStateSpace",
    "ModelAffineLPVLFR",
    "ModelRationalLPVLFR",
    "ModelNLLFR",
    "EnsembleLinStateSpace",
    "EnsembleAffineLPVLFR",
    "EnsembleRationalLPVLFR",
    "EnsembleNLLFR",
    "OptimConfig",
    "OptimHistory",
    "optimize",
    "optimize_ensemble",
    "iddata",
    "iddata_collection",
    "simulate_collection",
    "DataScalerConfig",
    "learn_initial_state",
    "SchedulingMapConfig",
    "StateEncoderConfig",
    "NetworkConfig",
    "save_model",
    "load_model",
    "scale_lti_ss",
    "metrics",
]
