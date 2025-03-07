#!/bin/bash

# Create & Activate python enviroment
if [ ! -d "gpytorch_env" ]; then
    python3 -m venv gpytorch_env
fi

# Activate enviroment
source gpytorch_env/bin/activate

# Install requirements
if ! python3 -c "import gpytorch"; then
    pip3 install gpytorch
fi

# Run the python script
python3 execute.py --use-gpu "$@"
