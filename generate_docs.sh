#!/bin/bash
jupyter nbconvert $1 --to html --output-dir=docs
