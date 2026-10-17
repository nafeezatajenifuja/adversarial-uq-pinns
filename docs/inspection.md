# Inspection and team provenance

## Source and inspection scope

The complete source tree at upstream commit `60da7db86b02426acf0a092af54891191645a0c5` contains two notebooks, one figure, the README, and seven PyCharm metadata files. All notebook source cells, saved output types, training records, recorded version output, relative-error output, IDE metadata, and the representative image were inspected before cleanup. The two notebooks contain identical model-class source.

The upstream repository exposes one original commit, authored by TaiwoAdebiyi23 on 10 December 2024. Team membership is recorded independently in the original README: Akash Yadav, Taiwo A. Adebiyi, and Nafeezat Ajenifuja. The author of the Git commit is not treated as the sole project author. This copy neither rewrites that commit nor invents individual contributions or coauthor trailers.

The source was cloned without a shallow-history option. The original remote was renamed from `origin` to `upstream`. Subsequent portfolio work is recorded as descendant commits. [The artifact manifest](../provenance/manifest.json) records the original commit, Git blob IDs, SHA-256 hashes, and preserved paths. [The upstream README](../provenance/upstream_README.md) is an unchanged archival snapshot; its original relative links reflect the earlier layout.

## Verified implementation

| Item | Verified behavior |
| --- | --- |
| Generator P | Input `(x,z)` of dimension 2; four 50-unit tanh hidden layers; one linear deflection output |
| Encoder Q | Input `(x,u)` of dimension 2; four 50-unit tanh hidden layers; one linear latent reconstruction |
| Discriminator T | Input `(x,u)` of dimension 2; two 50-unit tanh hidden layers; one linear score, transformed by sigmoid in its loss |
| Initialization | Xavier-scaled Gaussian weights and zero biases |
| Coordinate normalization | Mean and standard deviation from collocation coordinates; physical derivative correction through the normalization Jacobian |
| Latent distribution | Scalar standard-normal inputs, freshly sampled for data and collocation points each training iteration |
| Physics | Second-order residual `u_xx + 5*x`; no time input or fourth-order beam residual |
| Training | 30,000 configured iterations; all observations and collocation points used each iteration; one T update and five joint P/Q updates; Adam at 1e-4 |
| Regularization | lambda = 1.5 and beta = 1.0 |
| Prediction | 2,000 passes over 200 positions; fresh latent value for each position on each pass |
| Evaluation | Pointwise means and variances, mean ± 2 standard deviations, plots, and predictive-mean relative Euclidean error |
| Historical environment | Notebook metadata: Python 3.10.12; saved version cell: TensorFlow 2.16.1; NumPy/SciPy/Matplotlib versions are not recorded |

## Notebook mapping

All cell indices are zero-based in the preserved originals.

| Component | Boundary notebook | Material notebook |
| --- | --- | --- |
| Model class and loss functions | 1 | 2 |
| Imports / initial NumPy seed | 3 | 4 |
| TensorFlow version | 4 | 5 |
| Synthetic data and nominal reference | 5 | 6 |
| Network dimensions and regularization | 6 | 7 |
| Training invocation and saved log | 7 | 8 |
| Predictive sampling | 8 | 9 |
| Uncertainty figure | 9 | 10 |
| Relative error | 10 | 11 |

The boundary notebook prints a historical relative error of `7.912620e-03`; the material notebook prints `7.561304e-03`. These are saved, single-run predictive-mean comparisons, not newly reproduced benchmarks or distributional accuracy measures. The saved training logs each contain 300 progress lines, from iteration 0 through 29,900, consistent with logging every 100 steps in a 30,000-iteration loop. Saved execution counts are not strictly sequential, so the originals are retained as historical records rather than guaranteed execution traces.

## Cleanup changes

- Preserve the original README, both original notebooks including outputs, and the original root figure as exact upstream Git-blob bytes at documented paths.
- Extract the boundary notebook's saved PNG without replotting or changing its image bytes.
- Remove the seven `.idea` files from the current tracked tree; retain their original content in Git history.
- Add ignore rules, direct-dependency pins, a headless runner, cleaned notebooks, and technical documentation.
- Retain both original bibliography entries and correct the Goodfellow proceedings URL, whose upstream value returns HTTP 404.
- In cleaned copies only: remove outputs and commented-out exploratory blocks; add explanatory headings; omit the unused missing `plotting` import; introduce an explicit TensorFlow seed and fresh-graph setup; disable device-placement logging; remove an unused empty figure; correct the spelling of the plot's deflection label; save newly generated plots separately.

The original class computations, reference formulas, generated-data calculations, layer sizes, losses, optimizer rates, update ratio, training defaults, prediction defaults, and error formula are retained. No upstream license exists and no license has been added. This cleanup does not assign new ownership of the original scientific methods or teammates' contributions.
