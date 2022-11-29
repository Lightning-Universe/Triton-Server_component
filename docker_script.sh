mv /__model_artifacts/* /content
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
python /usr/local/bin/docker_script.py "$1"
