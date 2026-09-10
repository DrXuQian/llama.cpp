#!/usr/bin/env python3
"""Apply benchmark route controls in the profiled process, then exec it."""

import json
import os
import sys


CONTROLS = {"CUDA_VISIBLE_DEVICES", "PPU_SDK", "LD_LIBRARY_PATH", "PATH",
            "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION"}


def controlled(name):
    return name in CONTROLS or name.startswith(("QUACTLIZE_", "LLAMA_ARG_"))


def select(environment):
    return {k: v for k, v in environment.items() if controlled(k)}


def main():
    requested = json.loads(sys.argv[1])
    if not isinstance(requested, dict) or not all(
        isinstance(k, str) and controlled(k) and isinstance(v, str)
        for k, v in requested.items()
    ) or len(sys.argv) < 3:
        raise ValueError("invalid profiled application environment")
    # The collection service can outlive an earlier arm. Preserve profiler
    # injection variables, but never inherit that service's route controls.
    for key in list(os.environ):
        if controlled(key):
            del os.environ[key]
    os.environ.update(requested)
    print("KPACK_PROFILE_ENV " + json.dumps(select(os.environ), sort_keys=True), flush=True)
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
