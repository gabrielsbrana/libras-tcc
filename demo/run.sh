#!/usr/bin/env bash
# Sobe o servidor local da demonstração em http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
source "$HOME/libras-demo-venv/bin/activate"
exec uvicorn server.api_server:app --host 127.0.0.1 --port 8000
