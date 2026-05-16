#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "src/gco_hpif" ]; then
  echo "ERROR: Please run this script from the GCO-HPIF repository root."
  exit 1
fi

python scripts/phase1_docs_cleanup.py
