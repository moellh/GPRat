#!/bin/bash

################################################################################
# Run the GPXPy Python library on a test project
################################################################################

# Activate spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

if [ ! -d venv ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the python script
LD_PRELOAD=$(spack location -i gperftools)/lib/libtcmalloc.so python3 execute.py
