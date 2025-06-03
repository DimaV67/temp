#!/bin/bash
set -e  # Exit on error
set -x  # Print commands as they're executed

echo "Installing Poetry via pip..."
pip install poetry

echo "Checking Poetry version..."
poetry --version || echo "Poetry command not found in PATH"

echo "Finding Poetry installation path..."
find /usr -name poetry -type f -executable 2>/dev/null || echo "Poetry executable not found in /usr"
find ~/.local -name poetry -type f -executable 2>/dev/null || echo "Poetry executable not found in ~/.local"
python -m poetry --version || echo "Poetry module not directly runnable"

echo "Checking environment..."
echo "PATH: $PATH"
echo "Python site packages:"
python -c "import site; print(site.getsitepackages())"

echo "Trying to run poetry install..."
poetry install || echo "Poetry install failed"

echo "Trying to run poetry install with pip executable directly..."
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
echo "PATH after pipx: $PATH"
poetry install || echo "Poetry install still failed"