"""Execute a cleaned team notebook without requiring Jupyter.

Defaults retain the original 30,000 training iterations and 2,000 predictive
sampling passes. Smaller explicit counts are execution checks, not trained results.
"""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re


ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    """Reject empty or nonpositive run counts before constructing a TF graph."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("counts must be positive integers")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("boundary", "material"), required=True)
    parser.add_argument("--iterations", type=positive_integer)
    parser.add_argument("--samples", type=positive_integer)
    args = parser.parse_args()

    # These affect plotting/logging only. The notebook defines the numerical graph.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "matplotlib"))
    os.environ.setdefault("KERAS_HOME", str(ROOT / "results" / "keras"))
    os.chdir(ROOT)
    path = ROOT / "notebooks" / f"{args.experiment}_uncertainty.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}

    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if cell["id"] == "training" and args.iterations is not None:
                source, count = re.subn(r"nIter\s*=\s*30000", f"nIter = {args.iterations}", source)
                if count != 1:
                    raise ValueError("could not identify the notebook's training count")
            if cell["id"] == "prediction" and args.samples is not None:
                source, count = re.subn(r"N_samples\s*=\s*2000", f"N_samples = {args.samples}", source)
                if count != 1:
                    raise ValueError("could not identify the notebook's prediction count")
            exec(compile(source, f"{path.name}:{cell['id']}", "exec"), namespace)

        np = namespace["np"]
        samples = namespace["samples_mean"]
        expected = (namespace["N_ref"], namespace["N_samples"])
        if samples.shape != expected:
            raise ValueError(f"unexpected predictive array shape {samples.shape}; expected {expected}")
        for name in ("samples_mean", "mu_pred", "Sigma_pred", "error_u"):
            if not np.all(np.isfinite(namespace[name])):
                raise ValueError(f"nonfinite values in {name}")
        if np.any(namespace["Sigma_pred"] < 0):
            raise ValueError("negative predictive variance")

        model = namespace["model"]
        feed = {
            model.X_u_tf: model.X_u,
            model.Y_u_tf: model.Y_u,
            model.X_f_tf: model.X_f,
            model.Z_u_tf: np.zeros((len(model.X_u), model.Z_dim)),
            model.Z_f_tf: np.zeros((len(model.X_f), model.Z_dim)),
        }
        losses = model.sess.run([model.G_loss, model.T_loss], feed_dict=feed)
        if not np.all(np.isfinite(losses)):
            raise ValueError("nonfinite generator or discriminator loss")

        iterations = args.iterations if args.iterations is not None else 30000
        record = {
            "experiment": args.experiment,
            "iterations": iterations,
            "prediction_passes": namespace["N_samples"],
            "full_course_counts": iterations == 30000 and namespace["N_samples"] == 2000,
            "observation_shape": list(namespace["Y_u"].shape),
            "collocation_shape": list(namespace["X_f"].shape),
            "prediction_shape": list(samples.shape),
            "finite_predictions_and_losses": True,
            "relative_predictive_mean_error": float(namespace["error_u"]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: metadata.version(name) for name in ("tensorflow", "numpy", "scipy", "matplotlib")},
            "tensorflow_seed": 1234,
            "numpy_data_seed": 42,
            "figure": f"figures/generated/{args.experiment}_uncertainty.png",
        }
        output = ROOT / "results" / f"{args.experiment}_run.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Completed {args.experiment}: {iterations} iterations, predictive shape {samples.shape}.")
        print(f"Run record: {output.relative_to(ROOT)}")
    finally:
        if "model" in namespace:
            namespace["model"].sess.close()
        if "plt" in namespace:
            namespace["plt"].close("all")


if __name__ == "__main__":
    main()
