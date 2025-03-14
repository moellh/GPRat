#!/bin/bash

# Create & Activate python enviroment
if [ ! -d "gpflow_env" ]; then
    python3 -m venv gpflow_env
fi
source gpflow_env/bin/activate

# install requirements
if ! python3 -c "import gpflow"; then
    pip3 install gpflow
    pip3 install tensorflow[and-cuda]
fi

# Execute the python script
python3 execute.py "$@"
