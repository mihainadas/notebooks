"""Replace nbconvert's placeholder image text in reviewed notebook exports."""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path


DESCRIPTIONS = {
    "perceptron_en.html": (
        "Scatter plot of the linearly separable perceptron training data.",
        "Scatter plot of all points with the learned perceptron decision boundary.",
    ),
    "logistic_regression.html": (
        "Confusion matrix and ROC curve for the breast-cancer classification holdout set.",
    ),
}
PLACEHOLDER = "No description has been provided for this image"


def main() -> int:
    source = Path(sys.argv[1])
    path = Path(sys.argv[2])
    descriptions = DESCRIPTIONS.get(path.name, ())
    html = path.read_text(encoding="utf-8")
    for description in descriptions:
        html = html.replace(PLACEHOLDER, description, 1)
    if PLACEHOLDER in html:
        raise RuntimeError(f"Undescribed image remains in {path}")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    marker = f'<meta name="notebook-source-sha256" content="{source_digest}"/>'
    html = html.replace("</head>", f"{marker}\n</head>", 1)
    path.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
