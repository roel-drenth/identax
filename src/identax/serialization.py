"""Serialization utilities for saving and loading trained models.

This module provides functions to serialize models and ensembles to a single
zip file containing both the configuration (as JSON) and the weights (as
equinox binary format).

The config classes have been modified to accept strings for enum fields and
dicts for nested config fields, allowing direct JSON roundtrip.

Example usage:
    # Save a model
    from identax import save_model, ModelLinStateSpace
    save_model("my_model.zip", model)

    # Load a model
    from identax import load_model
    loaded = load_model("my_model.zip", ModelLinStateSpace)
"""

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax

from identax.models import (
    AbstractEnsemble,
    AffineLPVLFRConfig,
    EnsembleAffineLPVLFR,
    EnsembleLinStateSpace,
    EnsembleNLLFR,
    EnsembleRationalLPVLFR,
    LinStateSpaceConfig,
    ModelAffineLPVLFR,
    ModelLinStateSpace,
    ModelNLLFR,
    ModelRationalLPVLFR,
    NLLFRConfig,
    RationalLPVLFRConfig,
)


def save_model(path: str | Path, model: eqx.Module, overwrite: bool = False) -> None:
    """Save a model or ensemble to a single zip file.

    The zip file contains:
    - config.json: The model configuration as JSON
    - weights.eqx: The model weights in equinox binary format

    For ensembles, n_models is stored in the config JSON under a _n_models field.

    Parameters
    ----------
    path : str | Path
        Path to the output zip file.
    model : eqx.Module
        The model or ensemble to save.
    overwrite : bool, optional
        If True, overwrite existing file. If False (default), raise FileExistsError.

    Raises
    ------
    FileExistsError
        If the file already exists and overwrite is False.

    Examples
    --------
    >>> save_model("my_model.zip", model)
    >>> save_model("/path/to/ensemble.zip", ensemble)
    >>> save_model("model.zip", model, overwrite=True)  # Force overwrite
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. Use overwrite=True to replace."
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Save config as JSON
        config_dict = asdict(model.config)

        # Add metadata for ensembles
        if isinstance(model, AbstractEnsemble):
            config_dict["_n_models"] = model.n_models

        zf.writestr("config.json", json.dumps(config_dict, indent=2))

        # Save weights
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, model)
        zf.writestr("weights.eqx", buffer.getvalue())


def load_model(path: str | Path, model_class: type, **kwargs) -> eqx.Module:
    """Load a model or ensemble from a zip file.

    Parameters
    ----------
    path : str | Path
        Path to the zip file.
    model_class : type
        The model class (e.g., ModelLinStateSpace, EnsembleLinStateSpace).
    **kwargs : Any
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    eqx.Module
        The loaded model or ensemble.

    Examples
    --------
    >>> from identax import load_model, ModelLinStateSpace
    >>> loaded = load_model("my_model.zip", ModelLinStateSpace)
    """
    path = Path(path)

    with zipfile.ZipFile(path, "r") as zf:
        # Load config
        config_dict = json.loads(zf.read("config.json"))

        # Extract metadata
        n_models = config_dict.pop("_n_models", None)

        # Get the config class - use registry
        config_class = _CONFIG_MAP.get(model_class)
        if config_class is None:
            raise ValueError(f"Unknown model class: {model_class.__name__}")

        # Reconstruct config (config classes now accept strings/dicts)
        config = config_class(**config_dict)

        # Build template
        if issubclass(model_class, AbstractEnsemble):
            if n_models is None:
                raise ValueError("Ensemble requires _n_models in config")
            template = model_class(n_models, config, jax.random.PRNGKey(0), **kwargs)
        else:
            template = model_class(config, jax.random.PRNGKey(0), **kwargs)
            empty_scalers = template.scalers.from_shape(
                config.scalers,
                config_dict["nu"],
                config_dict["ny"],
                config_dict.get("nd", 0),
            )
            template = eqx.tree_at(
                lambda model: model.scalers, template, replace=empty_scalers
            )

        # Load weights
        buffer = io.BytesIO(zf.read("weights.eqx"))
        return eqx.tree_deserialise_leaves(buffer, template)


# Map model classes to their config classes
_CONFIG_MAP = {
    ModelLinStateSpace: LinStateSpaceConfig,
    ModelAffineLPVLFR: AffineLPVLFRConfig,
    ModelRationalLPVLFR: RationalLPVLFRConfig,
    ModelNLLFR: NLLFRConfig,
    EnsembleLinStateSpace: LinStateSpaceConfig,
    EnsembleAffineLPVLFR: AffineLPVLFRConfig,
    EnsembleRationalLPVLFR: RationalLPVLFRConfig,
    EnsembleNLLFR: NLLFRConfig,
}
