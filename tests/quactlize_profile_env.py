#!/usr/bin/env python3
"""Apply benchmark route controls in the profiled process, then exec it."""

import json
import os
import sys
from pathlib import Path


CONTROLS = {"CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PPU_SDK", "LD_LIBRARY_PATH", "PATH",
            "DG_JIT_HGCC_COMPILER", "DG_JIT_CACHE_DIR", "DG_LIBRARY_ROOT",
            "GGML_NCP_FA_LIB", "GGML_NCP_MOE_LIB", "GGML_NCP_GDN_LIB",
            "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION"}


def controlled(name):
    return name in CONTROLS or name.startswith(("QUACTLIZE_", "LLAMA_ARG_"))


def select(environment):
    return {k: v for k, v in environment.items() if controlled(k)}


def tool_environment(executable, environment):
    """Use the selected SDK for both the CLI and its service executables."""
    root = Path(executable).resolve().parent.parent
    result = dict(environment)
    result['ASIGHT_HOME'] = str(root)
    # FetchBinaryPath searches ASIGHT_LD_LIBRARY_PATH before PATH.
    result['ASIGHT_LD_LIBRARY_PATH'] = os.pathsep.join(map(str, (root / 'bin', root / 'lib')))
    result['PATH'] = str(root / 'bin') + os.pathsep + result.get('PATH', '')
    result['LD_LIBRARY_PATH'] = str(root / 'lib') + os.pathsep + result.get('LD_LIBRARY_PATH', '')
    return result


def service_snapshot(executable, environment):
    """Read service identity without stopping processes or removing locks."""
    result = dict(executable=str(Path(executable).resolve()),
                  environment={k: environment.get(k) for k in (
                      'ASIGHT_HOME', 'ASIGHT_LD_LIBRARY_PATH', 'PATH', 'LD_LIBRARY_PATH',
                      'PERFETTO_PRODUCER_SOCK_NAME', 'PERFETTO_CONSUMER_SOCK_NAME')}, services=[])
    for name in ('traced', 'traced_probe', 'traced_device', 'traced_perf'):
        record = dict(lock='/tmp/asight_' + name + '.pid')
        try:
            raw = Path(record['lock']).read_text().strip()
            if not raw.isdecimal():
                raise ValueError('non-numeric PID file')
            record['pid'] = int(raw)
            proc = Path('/proc') / raw
            record['executable'] = str((proc / 'exe').resolve(strict=True))
            record['profiler_libraries'] = sorted({line.split()[-1] for line in
                (proc / 'maps').read_text().splitlines() if '/libperfetto.so' in line or '/libhgpti.so' in line})
        except (OSError, ValueError) as error:
            record['unavailable'] = str(error)
        result['services'].append(record)
    return result


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
