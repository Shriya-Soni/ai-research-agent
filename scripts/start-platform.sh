#!/usr/bin/env bash
# Start all Research Intelligence Platform services
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

echo "Starting platform (5 services)..."
echo "Web UI will be at: http://localhost:8000"
echo ""

$PY -m rip.agents.web_retrieval &
$PY -m rip.agents.document_search &
$PY -m rip.agents.structured_data &
$PY -m rip.agents.synthesis &
sleep 2
$PY -m rip.api.main
