# Jupyter Notebook Documentation Generator

This project contains Jupyter notebooks focused on machine learning concepts, particularly the Perceptron algorithm, along with tools to generate HTML documentation from these notebooks.

## Project Structure

- `perceptron/` - Contains Jupyter notebooks about the Perceptron algorithm
  - `perceptron.ipynb` - Original notebook
  - `perceptron_improved.ipynb` - Enhanced version
  - `perceptron_translated.ipynb` - Translated version

- `docs/` - Generated HTML documentation
- `scratchpad/` - Experimental notebooks
- `generate_docs.sh` - Shell script to convert notebooks to HTML

## Requirements

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage
To convert a Jupyter notebook to HTML documentation:

`./generate_docs.sh notebook_path.ipynb`

The generated HTML will be saved in the `docs` directory.

## License
[MIT](LICENSE)