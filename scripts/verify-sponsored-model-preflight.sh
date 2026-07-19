#!/bin/sh
set -eu

stream_file=${1:?Usage: verify-sponsored-model-preflight.sh STREAM_FILE}

if grep -q '^event: error' "$stream_file"; then
  exit 1
fi

grep -q '^event: done' "$stream_file"
