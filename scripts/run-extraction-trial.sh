#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIAL="$REPO/target/target-speaker-extraction/app-trial"
export XDG_CONFIG_HOME="$TRIAL/config"
export XDG_CACHE_HOME="$TRIAL/cache"
export HUSHMIC_PLUGIN_SO="$TRIAL/bin/libdpdfnet_ladspa.so"
export HUSHMIC_MODEL_DIR="$REPO/assets/models"
export ORT_DYLIB_PATH="$REPO/assets/lib/libonnxruntime.so"
export HUSHMIC_TSE_PYTHON="$REPO/target/voice-reference-evaluation/venv/bin/python"
export HUSHMIC_TSE_SCRIPT="$TRIAL/scripts/experimental-live-extraction.py"
export HUSHMIC_TSE_REFERENCE="$TRIAL/config/hushmic/extraction-reference.bin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export HUSHMIC_NO_CPU_PIN=1
if [[ $# -gt 0 ]]; then
    exec "$TRIAL/bin/hushmic" "$@"
fi
"$TRIAL/bin/hushmic" quit >/dev/null 2>&1 || true
exec "$TRIAL/bin/hushmic" --tray
