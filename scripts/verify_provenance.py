"""Check archival integrity, original Git ancestry, and preserved numerical source.

These checks compare the cleaned numerical calculations against the actual
original notebooks. They do not evaluate trained predictive accuracy.
"""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def syntax(source: str) -> str:
    return ast.dump(ast.parse(source), include_attributes=False)


def source(notebook: dict, identifier: str) -> str:
    return "".join(next(c for c in notebook["cells"] if c["id"] == identifier)["source"])


def main() -> None:
    manifest = json.loads((ROOT / "provenance/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"] + manifest["extracted_outputs"]:
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("manifest path leaves the repository")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"preserved artifact changed: {item['path']}")
        if "git_blob" in item:
            blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
            if hashlib.sha1(blob).hexdigest() != item["git_blob"]:
                raise ValueError(f"upstream Git blob differs: {item['path']}")
    subprocess.run(
        ["git", "-c", "safe.directory=" + ROOT.as_posix(), "merge-base", "--is-ancestor",
         manifest["upstream_commit"], "HEAD"], cwd=ROOT, check=True,
    )

    for experiment, name, offset in [
        ("boundary", "UQPINNs_EX2_BC.ipynb", 0),
        ("material", "UQPINNs_EX2_E.ipynb", 1),
    ]:
        original = json.loads((ROOT / "notebooks/original" / name).read_bytes())
        cleaned = json.loads((ROOT / "notebooks" / f"{experiment}_uncertainty.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c.get("source", [])) for c in original["cells"]]
        original_class = "class ODE_UQPINN:" + cells[1 + offset].split("class ODE_UQPINN:", 1)[1]
        expected = {
            "model": original_class.replace("tf.ConfigProto(log_device_placement=True)",
                                            "tf.ConfigProto(log_device_placement=False)"),
            "data": cells[5 + offset],
            "configuration": cells[6 + offset],
            "training": cells[7 + offset],
            "prediction": cells[8 + offset].replace("plt.figure(1)\n", ""),
            "evaluation": cells[10 + offset],
        }
        for identifier, original_source in expected.items():
            if syntax(source(cleaned, identifier)) != syntax(original_source):
                raise ValueError(f"numerical source differs: {experiment}/{identifier}")
        expected_plot = cells[9 + offset].replace("Defelection", "Deflection")
        old_name = "UQ_with_GAN5.png" if experiment == "boundary" else "UQ_with_GAN4.png"
        expected_plot = expected_plot.replace(
            f"plt.savefig('{old_name}', dpi=300)",
            f'plt.savefig(OUTPUT_DIR / "{experiment}_uncertainty.png", dpi=300)',
        )
        if syntax(source(cleaned, "visualization")) != syntax(expected_plot):
            raise ValueError(f"plot calculations differ: {experiment}")
        ids = [c["id"] for c in cleaned["cells"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate notebook cell IDs")
        for cell in cleaned["cells"]:
            if cell["cell_type"] == "code":
                if cell["execution_count"] is not None or cell["outputs"]:
                    raise ValueError("cleaned notebook contains execution outputs")
                compile("".join(cell["source"]), cell["id"], "exec")

    print("Original Git ancestry and all preserved artifact hashes verified.")
    print("Both cleaned notebooks preserve model, data, configuration, training, prediction, and error calculations.")
    print("Plot differences are limited to the output path and the deflection-label spelling.")
    print("Cleaned notebook cells compile and contain no execution outputs.")


if __name__ == "__main__":
    main()
