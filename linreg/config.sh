PYTHON_PATHS="$(whereis python | sed 's/python: //')"
PYTHON=$(cut <<< $PYTHON_PATHS -d " " -f 1)
$PYTHON -m venv "$(pwd)"
source ./bin/activate
echo Virtual environnement has been configured
../bin/pip install numpy