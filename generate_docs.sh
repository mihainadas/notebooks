#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: ./generate_docs.sh notebook.ipynb output.html"
    exit 1
fi

notebook_path="$1"
output_name="$2"

if [ ! -f "$notebook_path" ]; then
    echo "Notebook not found: $notebook_path" >&2
    exit 1
fi

mkdir -p docs
jupyter nbconvert "$notebook_path" --to html --output-dir=docs --output="$output_name"
python scripts/add_image_descriptions.py "$notebook_path" "docs/$output_name"
