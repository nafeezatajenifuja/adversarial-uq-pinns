# Numerical interpretation and reproduction

## Executed physical model

The notebooks label their application as a cantilever-beam deflection example. The implemented reference is

```text
u_ref(x; E, I) = -50*x^3/(3*E*I)
E = 2e7, I = 1e-6, x in [0,1]
```

The generator's residual is `Jacobian**2 * u_XX + 5*x`, where `X` is the standardized coordinate and `x` is reconstructed in physical coordinates. This is equivalent to `u_xx + 5*x`; it is a second-order ODE, not a full fourth-order Euler–Bernoulli boundary-value formulation. The cubic reference satisfies this residual at the nominal E and I. The code does not establish a general beam solver or a verified set of physical load/unit conventions.

The preserved plot labels the vertical axis as deflection in `10^-4 m`, but the plotted arrays are not multiplied by `10^4`. For example, the unscaled nominal reference at x=1 is -5/6. The original image and its labels remain intact; the intended physical scaling should be confirmed with the team before quantitative reuse.

## Two observation models

The boundary experiment takes 20 standard-normal-driven offsets with mean 0 and standard deviation 5e-6. Within each endpoint pair, the same offset is subtracted at x=0 and x=1. There are 40 observed values, 200 collocation coordinates, and 200 reference positions.

The material experiment draws 20 modulus values from `Normal(2e7, 0.5e6)`. Each draw generates the cubic response at ten positions: 0, 0.5, and eight uniformly spaced positions from 0.6 to 1. This produces 200 observed values and uses 100 collocation coordinates and 200 reference positions. There is no positivity truncation of the Gaussian modulus distribution.

The material variation enters the synthetic observations only. The residual retains its nominal `+5*x` term; E is not supplied to the generator, encoder, or discriminator and is not inferred. This tension between uncertain data and nominal physics is part of the original experiment. A stochastic material-dependent residual or an inverse-modulus model would change the method and has not been introduced.

## Networks, losses, and training

There is one instance of each of three networks: P (generator/decoder), Q (encoder), and T (discriminator). P and Q each use four tanh hidden layers of width 50; T uses two. All linear outputs have dimension one. Weight initialization is Xavier-scaled Gaussian. Ymean and Ystd are computed by the original class but do not normalize its outputs.

With lambda=1.5 and beta=1, the minimized generator/encoder loss is

```text
mean(T(x, P(x,z)))
    + 0.5 * mean((z - Q(x, P(x,z)))^2)
    + mean(r(x,z)^2)
```

The code calls the first term `KL_loss`; it is the learned adversarial score, not an independently evaluated KL divergence. The collocation encoder expression is constructed but does not contribute to the minimized loss. The discriminator minimizes

```text
-mean(log(1 - sigmoid(T(x,u_observed)) + 1e-8)
      + log(sigmoid(T(x,P(x,z))) + 1e-8))
```

Both optimizers are Adam with learning rate 1e-4. Each of 30,000 configured iterations samples fresh observation and collocation latent arrays, then makes one discriminator and five joint generator/encoder updates. The class includes a minibatch helper, but these experiments use full batches.

## Interpreting predictive variability

Prediction calls the generator 2,000 times over 200 coordinates and calculates NumPy population variance (`ddof=0`). Each call draws a different scalar latent input at every coordinate, so a column of the sampled array is not a spatially coherent function draw. The plotted band is the pointwise mean plus or minus two sampled standard deviations. No Gaussian predictive-distribution assumption, simultaneous band, nominal 95% coverage, calibrated uncertainty, Bayesian weight posterior, or material-parameter posterior is established.

The relative-error cell compares the predictive mean with the nominal cubic reference using a relative Euclidean norm. It evaluates neither uncertainty calibration nor agreement with the full uncertain-data distribution. There is no separate held-out stochastic test set, baseline comparison, timing benchmark, or repeated-seed study.

## Dependencies and runtime

The preserved notebooks record Python 3.10.12 and TensorFlow 2.16.1 through the TensorFlow 1 compatibility API. NumPy, SciPy, and Matplotlib are imported but their historical versions are absent. `time`, `timeit`, `os`, and `pathlib` belong to Python's standard library. The imported `plotting.newfig/savefig` helper is absent upstream and neither function is used; the cleaned notebooks omit that import rather than adding an unrelated package.

The direct dependencies in `requirements.txt` are TensorFlow 2.16.1, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.8.4. Only the TensorFlow pin is recovered from historical output; the other pins define the portfolio validation environment. These versions are not presented as the original full environment. Python 3.10 is the recommended reproduction interpreter. TensorFlow's [official build table](https://www.tensorflow.org/install/source_windows) lists supported Python versions for 2.16.1; the host's Python 3.13 cannot install this pinned TensorFlow release.

## Environment setup

Create a Python 3.10 virtual environment, activate it, then install the requirements. For example, with Python 3.10 selected as `python`:

```bash
python -m venv .venv
```

In PowerShell, activate using `.\.venv\Scripts\Activate.ps1`; on macOS/Linux use `source .venv/bin/activate`. Alternatively, if uv is already installed, `uv venv --python 3.10 .venv` can select/download a compatible interpreter. uv is an optional environment tool, not a project dependency.

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py --experiment boundary
python scripts/run_experiment.py --experiment material
```

These commands retain the original training and prediction counts. They generate new seeded runs, not exact replays of the historical images. The original NumPy seed settings 1234 and then 42 are retained; the latter controls data generation followed by training and prediction draws. The cleaned copies additionally set TensorFlow's graph seed to 1234. TensorFlow initialization was not seeded upstream. Device placement, numerical libraries, and hardware can also affect numerical trajectories.

For an execution check:

```bash
python scripts/run_experiment.py --experiment boundary --iterations 2 --samples 8
python scripts/run_experiment.py --experiment material --iterations 2 --samples 8
```

Short runs check data preparation, graph construction, losses, updates, prediction shapes, finite values, and plotting. They do not reproduce a trained model or the saved relative errors. New PNGs go to ignored `figures/generated/`; machine-readable run records go to ignored `results/`. Historical figures and notebooks are not overwritten. The runner executes the same cleaned notebook cells and closes its session after execution.

## Provenance and validation

[Inspection notes](inspection.md) map scientific claims to the original cells. [The manifest](../provenance/manifest.json) records exact preserved artifacts; [the original team README](../provenance/upstream_README.md) retains the team and bibliography. Course provenance and the three authors remain explicit in the main README.

[The validation record](validation.json) documents successful isolated installation on Windows with Python 3.10.20, compatible installed dependencies, preserved numerical-source checks, and both two-iteration/eight-sampling-pass execution checks. It does not claim full-training reproduction. Use `python scripts/verify_provenance.py` to repeat artifact-hash, Git-ancestry, source-equivalence, and cleaned-notebook checks without training.
