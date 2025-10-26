#!/usr/bin/env bash

set -euo pipefail

LEVEL_ARG="${1:-4}"
# Accept either pure numeric (e.g. "3") or the fully-qualified form (e.g. "L3").
LEVEL_NUM="${LEVEL_ARG#L}"
if [[ -z "${LEVEL_NUM}" ]]; then
  echo "error: expected a pyramid level (e.g. 4 or L4)" >&2
  exit 1
fi

LEVEL="L${LEVEL_NUM}"

./target/debug/segment_refine_demo ./config/segment_refine_demo.json

if [[ -f ../venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../venv/bin/activate
fi

python tools/plot_coarse_segments.py \
  --image "out/segment_demo/pyramid_${LEVEL}.png" \
  --segments "out/segment_demo/segments_${LEVEL}.json"
