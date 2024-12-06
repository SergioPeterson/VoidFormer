#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
echo "Script directory: $SCRIPT_DIR"

# Initialize conda in the script's environment
eval "$(conda shell.bash hook)"

# Check if "voidFormer" environment already exists
if conda env list | grep -qw 'voidFormer'; then
    echo "Conda environment 'voidFormer' already exists. Skipping creation. Activating pokerenv."
else
    echo "Creating conda environment 'voidFormer'."
    conda create --name voidFormer python=3.10 -y
fi

conda activate voidFormer

# Install packages if not already installed
echo "Installing packages..."
# conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio -c pytorch -c nvidia -y
# conda install transformers pandas numpy=1.26.4 tqdm scikit-learn -y

# Install additional Python packages from requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing pip packages from requirements.txt"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping pip installations."
fi

echo "Environment setup complete."