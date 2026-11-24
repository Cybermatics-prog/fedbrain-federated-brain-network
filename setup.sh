#!/usr/bin/env sh
# SPDX-License-Identifier: MIT
# Convenience wrapper. The canonical entry point is `python scripts/pipeline.py setup`.
set -eu
if command -v python3 >/dev/null 2>&1; then
    exec python3 scripts/pipeline.py setup "$@"
fi
exec python scripts/pipeline.py setup "$@"
