# moving model code form the mounted volume; we do this to overcome the mount volume delay in mac os
cp -r /__model_artifacts/* /content

# install dependencies if requirements.txt is present
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# run the setup file - install the shebang-like commands
python /usr/local/bin/docker_script.py "$1"

# install lightning_triton if not installed already
if ! python -c "import lightning_triton" &> /dev/null; then
    pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
fi
