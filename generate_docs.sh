#!/bin/bash
# Usage: ./generate_docs.sh notebook.ipynb output.html
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./generate_docs.sh notebook.ipynb output.html"
    exit 1
fi
jupyter nbconvert $1 --to html --output-dir=docs --output=$2
