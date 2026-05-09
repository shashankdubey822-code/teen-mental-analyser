#!/bin/bash
set -e

echo "==> Bypassing Render Default Python (Currently cached to 3.14)"
echo "==> Installing isolated Python 3.11.8 environment..."

# Download and extract Python 3.11.8 (Linux x86_64) from official python-build-standalone
curl -L -o python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240224/cpython-3.11.8+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar -xzf python.tar.gz -C /opt/
rm python.tar.gz

export PATH="/opt/python/bin:$PATH"
export PYTHON_VERSION="3.11.8"

echo "==> Python Version verification:"
python --version

echo "==> Installing dependencies using strictly binary wheels where possible..."
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --only-binary :all: pydantic-core pydantic fastapi starlette || true
python -m pip install --no-cache-dir -r requirements.txt

echo "==> Build complete. Environment is locked to Python 3.11."
