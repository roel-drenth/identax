<h1 align="center">Identax</h1>

Identax is a JAX toolbox for efficient learning of linear, linear parameter-varying (LPV) and nonlinear dynamical systems. It aims to be a transparent, modular, scalable and efficient foundation for system identification research.

The first release provides:

- **Model structures**: linear time-invariant state-space (LTI-SS) models, LPV models in linear fractional representation (LPV-LFR) with affine or rational scheduling dependency, and nonlinear LFR (NL-LFR) models. LPV and nonlinear models can be combined with a subspace encoder for initial state estimation.
- **Identification criteria**: prediction error minimization (PEM) on one or multiple experiments, with optional ℓ2-regularization, and multiple-shooting criteria (unconstrained multiple shooting, UMS, and encoder-based multiple shooting, EBMS) for parallel training on CPU and GPU.
- **Transparent optimization**: a two-step Adam → L-BFGS procedure built on Optax, with convergence-based stopping, early stopping on validation data, and the full training and validation loss history of both steps.
- **Usability**: data objects for single and multiple experiments, automatic data scaling, ensembles of models with different initializations, and saving and loading of models.

Identax is built on [JAX](https://github.com/jax-ml/jax), [Equinox](https://github.com/patrick-kidger/equinox) and [Optax](https://github.com/google-deepmind/optax).

## Installation

Identax requires Python 3.12 or newer. Install the latest version directly from GitHub:

```bash
uv venv
uv pip install git+https://github.com/roel-drenth/identax.git
```

Optional extras:

```bash
# Dependencies of the example notebooks (matplotlib, scipy, nonlinear-benchmarks)
uv pip install "identax[examples] @ git+https://github.com/roel-drenth/identax.git"

# GPU support, pick one: CUDA 13 (NVIDIA driver >= 580) or CUDA 12 (driver >= 525)
uv pip install "identax[cuda] @ git+https://github.com/roel-drenth/identax.git"
uv pip install "identax[cuda12] @ git+https://github.com/roel-drenth/identax.git"
```

> **Note:** importing `identax` enables 64-bit precision in JAX (`jax_enable_x64`) for the whole Python process.

## Quickstart

Identify an LTI-SS model, estimate the initial state on test data and evaluate the simulation error:

```python
import jax
import jax.numpy as jnp
import identax as idx

# Experiment data: U of shape (N, nu), Y of shape (N, ny)
train_data = idx.iddata(U_train, Y_train)
val_data = idx.iddata(U_val, Y_val)
test_data = idx.iddata(U_test, Y_test)

# Model, created from its configuration
key = jax.random.PRNGKey(42)  # seed for initialization
lti_cfg = idx.LinStateSpaceConfig(nx=2, nu=1, ny=1)
lti_mod = idx.ModelLinStateSpace(lti_cfg, key)

# Adam warm start followed by L-BFGS, with early stopping on the validation data
optim_cfg = idx.OptimConfig(adam_epochs=1000, lbfgs_epochs=1000)
lti_mod, x0, history = idx.optimize(
    lti_mod, train_data, jnp.zeros(2), optim_cfg, val_data=val_data
)

# Estimate the initial state from the first 50 test samples, then simulate
x0_test, _ = idx.learn_initial_state(lti_mod, test_data, jnp.zeros(2), lbfgs_epochs=100, T=50)
y_hat, x_hat = lti_mod.simulate(test_data, x0_test)
print("Test RMSE:", idx.metrics.RMSE(test_data.Y[50:], y_hat[50:]))
```

Data is passed in physical units. Each model holds a `DataScaler`, which by default standardizes the signals using the training data on the first call to `optimize()`, so you don't need to scale data yourself.

`history` is an `OptimHistory` with the training and validation loss of both optimization steps, the training times and the reason L-BFGS stopped.

## Model structures

**LPV-LFR models** have a separate configuration for their scheduling map, here a ResNet:

```python
sched_cfg = idx.SchedulingMapConfig(network="ResNet", layers=2, neurons_per_layer=6)
lpv_cfg = idx.AffineLPVLFRConfig(
    nx=2, nu=1, ny=1,
    npar=1,   # number of scheduling signals
    nreps=1,  # number of repetitions in Delta
    nd=0,     # no exogenous scheduling signals
    SchedulingMap=sched_cfg,
)
lpv_mod = idx.ModelAffineLPVLFR(lpv_cfg, key)
```

Use `RationalLPVLFRConfig` and `ModelRationalLPVLFR` for rational scheduling dependency. With `nd > 0`, pass the exogenous scheduling signals as `idx.iddata(U, Y, D=D)`, with `D` of shape (N, nd).

**NL-LFR models**, here with a subspace encoder for initial state estimation:

```python
nllfr_cfg = idx.NLLFRConfig(
    nx=2, nu=1, ny=1, nw=1, nz=1,
    DeltaMap=idx.NetworkConfig(neurons_per_layer=6),  # feedback nonlinearity
    StateEncoder=idx.StateEncoderConfig(n_hist=5),
)
nllfr_mod = idx.ModelNLLFR(nllfr_cfg, key)
```

**Ensembles** hold several models with the same structure but different initializations, trained in parallel with `optimize_ensemble()`:

```python
nllfr_ens = idx.EnsembleNLLFR(n_models=10, config=nllfr_cfg, key=key)
nllfr_ens, x0, history = idx.optimize_ensemble(nllfr_ens, train_data, jnp.zeros(2), optim_cfg)
```

## Multiple shooting and multiple experiments

Multiple shooting splits the data into sub-records of length `T`, which are simulated in parallel. Choose unconstrained multiple shooting with `shooting_type="multiple"`, or encoder-based multiple shooting with `shooting_type="encoder"` (requires a model with a `StateEncoder`):

```python
optim_cfg = idx.OptimConfig(
    adam_epochs=1000, lbfgs_epochs=2000,
    shooting_type="encoder",  # use EBMS
    T=200, overlap=20,        # sub-record length and overlap
    l2_rho=1e-6,              # l2-regularization weight
)
nllfr_mod, _, history = idx.optimize(nllfr_mod, train_data, jnp.zeros(2), optim_cfg)
```

To train on several experiments, possibly of different lengths, pass an `iddata_collection`:

```python
multi_exp = idx.iddata_collection([
    idx.iddata(U1, Y1),
    idx.iddata(U2, Y2),
])
optim_cfg = idx.OptimConfig(adam_epochs=1000, lbfgs_epochs=1000)
lti_mod, x0s, history = idx.optimize(lti_mod, multi_exp, jnp.zeros(2), optim_cfg)
```

`x0s` then holds one learned initial state per experiment.

## Saving and loading

```python
idx.save_model("lti_model.zip", lti_mod, overwrite=True)
loaded = idx.load_model("lti_model.zip", idx.ModelLinStateSpace)
```

The zip file contains the model configuration (`config.json`) and the parameters (`weights.eqx`). Loading requires the model class.

## Examples

The [Examples](Examples) folder contains notebooks for the benchmarks in the paper, the Silverbox and a control moment gyroscope.

To run them, clone the repository and install Identax with the `examples` extra in a local environment:

```bash
git clone https://github.com/roel-drenth/identax.git
cd identax
uv sync --extra examples
```

This creates a `.venv` with the versions pinned in `uv.lock`. Add `--extra cuda` (or `--extra cuda12`) to train on a GPU. Then start Jupyter in that environment and open a notebook from the `Examples` folder:

```bash
uv run --with jupyter jupyter lab Examples
```

Alternatively, open the notebooks in an editor such as VS Code and select `.venv` as the kernel.
