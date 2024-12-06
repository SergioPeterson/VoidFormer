#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
echo "Script directory: $SCRIPT_DIR"

# Initialize conda in the script's environment
eval "$(conda shell.bash hook)"

# Check if "voidFormer" environment already exists
if conda env list | grep -qw 'voidFormer'; then
    echo "Conda environment 'voidFormer' already exists. Skipping creation. Activating voidFormer."
else
    echo "Creating conda environment 'voidFormer'."
    conda create --name voidFormer python=3.9 -y
fi

conda activate voidFormer

# Install packages if not already installed
echo "Installing packages..."

# Install additional Python packages from requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing pip packages from requirements.txt using PyTorch CUDA index URL"
    pip install -r "$SCRIPT_DIR/requirements.txt" --index-url https://download.pytorch.org/whl/cu121
else
    echo "No requirements.txt found. Skipping pip installations."
fi

echo "Environment setup complete."
