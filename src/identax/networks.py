"""Neural networks used inside the models.

This module provides:

- ``SchedulingMapConfig`` and ``NNSchedulingMap``: the network that maps
  state, input and (optional) exogenous signal to the scheduling variable
  of the LPV-LFR models.
- ``StateEncoderConfig`` and ``NNStateEncoder``: the network that estimates
  the initial state from a window of past data, for encoder based multiple-shooting.
- ``NetworkConfig``: configuration for the static nonlinearity
  ``w = Delta(z)`` of NL-LFR models.
- ``ResNet``: an MLP with a linear skip connection, the default network type.
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray

_vmap_a_of_b = eqx.filter_vmap(lambda a, b: a(b))


def xavier_reinit_mlp(mlp: eqx.Module, key: PRNGKeyArray) -> eqx.Module:
    """Re-initialize the weights of all linear layers with Glorot (Xavier) uniform.

    Biases are left unchanged.

    Parameters
    ----------
    mlp : eqx.Module
        Network containing ``eqx.nn.Linear`` layers, e.g. an ``eqx.nn.MLP``
        or a ``ResNet``.
    key : PRNGKeyArray
        Random key.

    Returns
    -------
    eqx.Module
        The network with new weights.
    """

    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(mlp):
        return [
            x.weight
            for x in jax.tree_util.tree_leaves(mlp, is_leaf=is_linear)
            if is_linear(x)
        ]

    weights = get_weights(mlp)
    keys = jax.random.split(key, len(weights))

    glorot = jax.nn.initializers.glorot_uniform()

    new_weights = [glorot(k, w.shape, w.dtype) for w, k in zip(weights, keys)]

    return eqx.tree_at(get_weights, mlp, new_weights)


class Network(StrEnum):
    """Available network architectures.

    - ``RESNET``: MLP with a linear skip connection from input to output
      (see ``ResNet``).
    - ``MLP``: plain multilayer perceptron (``eqx.nn.MLP``).
    """

    RESNET = "ResNet"
    MLP = "MLP"


class Activation(StrEnum):
    """Available activation functions for the hidden layers."""

    RELU = "relu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"


@dataclass(frozen=True, kw_only=True)
class SchedulingMapConfig:
    """Configuration for scheduling maps used with LPV models

    Parameters
    ----------
    network             : Network | str
                        Type of network used in scheduling map.
    layers              : int
                        Number of hidden-layers
    neurons_per_layer   : int
                        Number of neurons per layer.
    clip                : bool
                        Is the output clipped between -1 and 1?
    activation          : Activation | str
                        Type of activation function used.

    """

    network: Network | str = Network.RESNET
    layers: int = 2
    neurons_per_layer: int = 12
    clip: bool = True
    activation: Activation | str = eqx.field(static=True, default=Activation.TANH)

    def __post_init__(self) -> None:
        if isinstance(self.network, str):
            object.__setattr__(self, "network", Network(self.network))
        else:
            Network(self.network)
        if self.layers < 0:
            raise ValueError(f"layers must be >= 0, got {self.layers}")
        if self.neurons_per_layer <= 0:
            raise ValueError(
                f"neurons_per_layer must be > 0, got {self.neurons_per_layer}"
            )
        if isinstance(self.activation, str):
            object.__setattr__(self, "activation", Activation(self.activation))
        else:
            Activation(self.activation)


@dataclass(frozen=True, kw_only=True)
class StateEncoderConfig:
    """Configuration for encoder network used for encoder-based multiple-shooting.

    Parameters
    ----------
    network             : Network | str
                        Type of network used in encoder network.
    n_hist              : int
                        Number of past samples of u, y (and d) used to
                        estimate the initial state. Required.
    layers              : int
                        Number of hidden-layers
    neurons_per_layer   : int
                        Number of neurons per layer.
    activation          : Activation | str
                        Type of activation function used.
    clip                : bool
                        Is the output clipped between -1 and 1?

    """

    network: Network | str = Network.RESNET
    n_hist: int
    layers: int = 2
    neurons_per_layer: int = 16
    activation: Activation | str = eqx.field(static=True, default=Activation.TANH)
    clip: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.network, str):
            object.__setattr__(self, "network", Network(self.network))
        else:
            Network(self.network)
        if self.n_hist <= 0:
            raise ValueError(f"n_hist must be > 0, got {self.n_hist}")
        if self.layers < 0:
            raise ValueError(f"layers must be >= 0, got {self.layers}")
        if self.neurons_per_layer <= 0:
            raise ValueError(
                f"neurons_per_layer must be > 0, got {self.neurons_per_layer}"
            )
        if isinstance(self.activation, str):
            object.__setattr__(self, "activation", Activation(self.activation))
        else:
            Activation(self.activation)


@dataclass(frozen=True, kw_only=True)
class NetworkConfig:
    """Configuration for neural networks.

    Parameters
    ----------
    network             : Network | str
                        Type of network.
    layers              : int
                        Number of hidden-layers.
    neurons_per_layer   : int
                        Number of neurons per layer.
    activation          : Activation | str
                        Type of activation function used.
    clip                : bool
                        Is the output clipped between -1 and 1?

    """

    network: Network | str = Network.RESNET
    layers: int = 2
    neurons_per_layer: int = 16
    activation: Activation | str = eqx.field(static=True, default=Activation.TANH)
    clip: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.network, str):
            object.__setattr__(self, "network", Network(self.network))
        else:
            Network(self.network)
        if self.layers < 0:
            raise ValueError(f"layers must be >= 0, got {self.layers}")
        if self.neurons_per_layer <= 0:
            raise ValueError(
                f"neurons_per_layer must be > 0, got {self.neurons_per_layer}"
            )
        if isinstance(self.activation, str):
            object.__setattr__(self, "activation", Activation(self.activation))
        else:
            Activation(self.activation)


class ResNet(eqx.Module):
    """MLP with a linear skip connection: ``f(x) = act(MLP(x) + W x + b)``.

    The skip connection lets the network represent a linear map exactly,
    so the nonlinear part only has to model the deviation from it. The
    constructor follows the signature of ``eqx.nn.MLP``.

    Parameters
    ----------
    in_size : int | "scalar"
        Input dimension, or ``"scalar"`` for a scalar input.
    out_size : int | "scalar"
        Output dimension, or ``"scalar"`` for a scalar output.
    width_size : int
        Number of neurons per hidden layer.
    depth : int
        Number of hidden layers of the MLP part.
    activation : Callable
        Activation function of the hidden layers.
    final_activation : Callable
        Activation applied to the sum of the MLP and skip connection.
    key : PRNGKeyArray
        Random key for initialization.
    """

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)
    MLP: eqx.nn.MLP
    Residual: eqx.nn.Linear

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = jax.nn.tanh,
        *,
        key: PRNGKeyArray,
    ):

        Residkey, MLPkey = jax.random.split(key, 2)

        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

        self.Residual = eqx.nn.Linear(in_size, out_size, use_bias=True, key=Residkey)

        self.MLP = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=MLPkey,
        )

        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()

    def __call__(self, input):
        if self.out_size == "scalar":
            output = self.final_activation(self.MLP(input) + self.Residual(input))
        else:
            output = _vmap_a_of_b(
                self.final_activation, self.MLP(input) + self.Residual(input)
            )
        return output


class AbstractSchedulingMap(eqx.Module):
    """Interface for scheduling maps ``p = psi(x, u, d)`` of LPV models.

    Subclasses implement ``_evaluate``, which maps a single sample of the
    state x (nx,), input u (nu,) and exogenous signal d (nd,) to the
    scheduling variable p (npar,).
    """

    @abstractmethod
    def _evaluate(self, x: Array, u: Array, d: Array) -> Array:
        raise NotImplementedError


class NNSchedulingMap(AbstractSchedulingMap):
    """Scheduling map given by a neural network on ``[x, u, d]``.

    Weights are initialized with Glorot uniform (see ``xavier_reinit_mlp``).
    With ``config.clip``, the output passes through tanh, so that p stays in
    (-1, 1).

    Parameters
    ----------
    config : SchedulingMapConfig
        Network settings.
    in_size : int
        Network input size, ``nx + nu + nd``.
    out_size : int
        Number of scheduling variables ``npar``.
    key : PRNGKeyArray
        Random key for initialization.
    """

    layers: int = eqx.field(static=True)
    neurons_per_layer: int = eqx.field(static=True)
    clip: bool = eqx.field(static=True)

    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)
    network: eqx.Module

    def __init__(
        self,
        config: SchedulingMapConfig,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
    ):
        self.layers = config.layers
        self.neurons_per_layer = config.neurons_per_layer
        self.activation = config.activation
        self.clip = config.clip
        self.final_activation = jax.nn.tanh if config.clip else (lambda x: x)

        network = NETWORKS[config.network](
            in_size=in_size,
            out_size=out_size,
            width_size=config.neurons_per_layer,
            depth=config.layers,
            activation=ACTIVATIONS[config.activation],
            key=key,
            final_activation=self.final_activation,
        )
        self.network = xavier_reinit_mlp(network, key)

    def _evaluate(self, x: Array, u: Array, d: Array) -> Array:
        parts = [x, u]

        if d is not None and jnp.size(d) > 0:
            parts.append(d)
        input = jnp.concatenate(parts, axis=0)
        return self.network(input)


class NNStateEncoder(eqx.Module):
    """Neural network state encoder, estimates the state following a window
    of n_hist past outputs, inputs, and (optionally) exogenous signals.

    Parameters
    ----------
    config : StateEncoderConfig
        Network settings, including ``n_hist``.
    nx, nu, ny, nd : int
        State, input, output and exogenous signal dimensions.
    key : PRNGKeyArray
        Random key for initialization.

    Attributes
    ----------
    network : eqx.Module
            Network mapping the flattened history window to the state.
    n_hist  : int
            Number of past samples used for state estimation.
    nd      : int
            Number of exogenous signals included in the history window.
    """

    network: eqx.Module
    n_hist: int = eqx.field(static=True)
    nd: int = eqx.field(static=True)

    def __init__(
        self,
        config: StateEncoderConfig,
        nx: int,
        nu: int,
        ny: int,
        nd: int,
        key: PRNGKeyArray,
    ):
        self.n_hist = config.n_hist
        self.nd = nd
        self.network = NETWORKS[config.network](
            in_size=(nu + ny + nd) * config.n_hist,
            out_size=nx,
            width_size=config.neurons_per_layer,
            depth=config.layers,
            activation=ACTIVATIONS[config.activation],
            final_activation=jax.nn.tanh if config.clip else (lambda x: x),
            key=key,
        )

    def _inputs(self, data) -> Array:
        """Flatten the history window of data into the network input."""
        hist = [data.Y_hist, data.U_hist]
        if self.nd > 0:
            hist.append(data.D_hist)
        return jnp.concatenate(hist, axis=-1).reshape(-1)

    def __call__(self, data) -> Array:
        """Estimate the state following a single history window.

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
        return self.network(self._inputs(data))


NETWORKS = {
    Network.RESNET: ResNet,
    Network.MLP: eqx.nn.MLP,
}


ACTIVATIONS = {
    Activation.RELU: jax.nn.relu,
    Activation.SWISH: jax.nn.swish,
    Activation.TANH: jax.nn.tanh,
    Activation.SIGMOID: jax.nn.sigmoid,
    Activation.SOFTPLUS: jax.nn.softplus,
    Activation.IDENTITY: jax.nn.identity,
}
