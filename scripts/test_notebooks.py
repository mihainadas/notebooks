"""Validate all notebooks and execute the curated offline set."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
CURATED = (
    ROOT / "perceptron" / "perceptron.ipynb",
    ROOT / "numpy" / "numpy_vs_python.ipynb",
    ROOT / "logistic_regression" / "logistic_regression.ipynb",
)
RENDERED = {
    ROOT / "perceptron" / "perceptron.ipynb": ROOT / "docs" / "perceptron_en.html",
    ROOT / "numpy" / "numpy_vs_python.ipynb": ROOT / "docs" / "numpy_vs_python.html",
    ROOT / "logistic_regression" / "logistic_regression.ipynb": ROOT / "docs" / "logistic_regression.html",
}
DOC_MANIFEST = ROOT / "docs" / "notebook_sources.sha256"
UNSAFE_OUTPUT_MARKERS = ("Traceback (most recent call last)", "/Users/", "/private/tmp/")


def validate_all() -> None:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    if not notebooks:
        raise RuntimeError("No notebooks found")

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f"validated {path.relative_to(ROOT)}")


def verify_saved_outputs() -> None:
    for path in sorted(ROOT.rglob("*.ipynb")):
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "scratchpad":
            continue
        notebook = nbformat.read(path, as_version=4)
        for cell_index, cell in enumerate(notebook.cells):
            for output in cell.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "error":
                    raise RuntimeError(f"{relative} cell {cell_index} has a saved error")
                if output_type == "stream" and output.get("name") == "stderr":
                    raise RuntimeError(f"{relative} cell {cell_index} has saved stderr")
                rendered = str(output)
                marker = next((item for item in UNSAFE_OUTPUT_MARKERS if item in rendered), None)
                if marker:
                    raise RuntimeError(
                        f"{relative} cell {cell_index} contains unsafe saved output: {marker}"
                    )
        print(f"outputs ok {relative}")


def verify_docs_manifest() -> None:
    expected = {}
    for line in DOC_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative] = digest
    for source in CURATED:
        relative = source.relative_to(ROOT).as_posix()
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected.get(relative) != actual:
            raise RuntimeError(f"Rendered docs are stale for {relative}")
        html = RENDERED[source].read_text(encoding="utf-8")
        marker = f'<meta name="notebook-source-sha256" content="{actual}"/>'
        if marker not in html:
            raise RuntimeError(f"Rendered HTML is stale for {relative}")
        print(f"docs current {relative}")


def execute_curated() -> None:
    with tempfile.TemporaryDirectory(prefix="notebook-tests-") as temporary:
        test_root = Path(temporary)
        for source in CURATED:
            target_dir = test_root / source.parent.name
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            shutil.copy2(source, target)

            notebook = nbformat.read(target, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=180,
                kernel_name="python3",
                resources={"metadata": {"path": str(target_dir)}},
            )
            executed = client.execute()

            stderr = []
            for cell in executed.cells:
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "stream" and output.get("name") == "stderr":
                        stderr.append(output.get("text", ""))
            if stderr:
                joined = "\n".join(stderr)
                raise RuntimeError(f"{source.relative_to(ROOT)} wrote to stderr:\n{joined}")

            print(f"executed  {source.relative_to(ROOT)}")


def main() -> int:
    validate_all()
    verify_saved_outputs()
    verify_docs_manifest()
    execute_curated()
    return 0


if __name__ == "__main__":
    sys.exit(main())
