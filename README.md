# Machine learning notebooks

Small, inspectable notebooks for foundational machine-learning ideas and research-side data analysis. The emphasis is on code that can be executed from top to bottom, with claims tied to saved outputs.

## Curated notebooks

| Notebook | Question | Status |
| --- | --- | --- |
| [`perceptron/perceptron.ipynb`](perceptron/perceptron.ipynb) | How does a perceptron update a separator over two input features? | Executable tutorial |
| [`numpy/numpy_vs_python.ipynb`](numpy/numpy_vs_python.ipynb) | How should a NumPy-versus-loop microbenchmark be measured? | Executable benchmark |
| [`logistic_regression/logistic_regression.ipynb`](logistic_regression/logistic_regression.ipynb) | What does a defensible tabular-classification baseline look like? | Executable baseline |

The `scratchpad/` directory contains experiments that may require external model downloads or retired APIs. It is not part of the tested set and should not be cited as a reproducible result. Earlier survey analyses and generated perceptron variants are retained with outputs cleared under `archive/`; neither is current work.

## Run locally

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/test_notebooks.py
```

The test runner validates every notebook file, checks saved outputs outside `scratchpad/`, verifies the rendered-document source manifest, and executes the three curated, offline notebooks in isolated temporary copies. It fails on execution errors and warnings written to `stderr`.

To render one notebook for GitHub Pages:

```sh
./generate_docs.sh perceptron/perceptron.ipynb perceptron_en.html
```

Generated HTML is written to `docs/`.

## Reproducibility policy

- Saved outputs must come from a clean top-to-bottom run.
- Benchmarks report repeated measurements and the runtime environment; a single timing is not a result.
- Randomized examples use fixed local generators rather than global state.
- Warnings are treated as defects in the curated notebooks.
- Research notebooks state their population and denominators; exploratory work stays in `scratchpad/` until it meets the same bar.

## License

[MIT](LICENSE)
