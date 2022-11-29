import sys
from lightning.app.utilities.app_commands import run_app_commands
entrypoint_file = sys.argv[1]
if not entrypoint_file.endswith(".py"):
    raise ValueError("The entrypoint doesn't seem to be a python file.")
run_app_commands(entrypoint_file)


