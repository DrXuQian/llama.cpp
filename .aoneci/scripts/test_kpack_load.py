#!/usr/bin/env python3
"""Compare the actual buffer's GPU shuffle with offline reference GGUF bytes."""

import argparse
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

GGUF_PY = str(Path(__file__).resolve().parents[2] / 'gguf-py')
sys.path.insert(0, GGUF_PY)
import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='offline gguf_kpack.py from ncp_flash_lib/third_party/quactlize')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--packer', type=Path, default=os.environ.get('QUACTLIZE_PPU_PACK_LIBRARY'),
                        help='GPU pack library; defaults to QUACTLIZE_PPU_PACK_LIBRARY or the library beside --binary')
    parser.add_argument('--output', type=Path,
                        help='new directory; default: temporary kpack-load-* directory, removed on success')
    parser.add_argument('--backend', default='CUDA0')
    args = parser.parse_args()
    if args.packer is None:
        args.packer = args.binary.resolve().parent / 'libquactlize_ppu_pack.so'
    for path in (args.binary, args.packer, args.reference):
        if not path.is_file():
            parser.error(f'missing {path}')
    temporary = args.output is None
    if temporary:
        args.output = Path(tempfile.mkdtemp(prefix='kpack-load-', dir=Path.cwd()))
    else:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f'--output must be a new directory, {args.output} already exists')
    args.output = args.output.resolve()
    run_suite(args)
    if temporary:
        shutil.rmtree(args.output)
        print(f'KPACK_LOAD_CLEANUP removed={args.output}', flush=True)


def run_suite(args):
    print(f'KPACK_LOAD_START results={args.output}', flush=True)
    source, packed = args.output / 'source.gguf', args.output / 'reference.gguf'
    writer = gguf.GGUFWriter(source, 'llama')
    for q in range(10, 15):
        block, block_bytes = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(q)]
        for experts in (1, 3):
            shape = (experts, 256, 512 // block * block_bytes)
            raw = np.random.default_rng(q * 10 + experts).integers(0, 256, shape, dtype=np.uint8)
            writer.add_tensor(f'q{q}.e{experts}.weight', raw, raw_dtype=gguf.GGMLQuantizationType(q))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reference = args.reference.resolve()
    env = os.environ | {'QUACTLIZE_PPU_PACK_LIBRARY': str(args.packer.resolve()),
                        'PYTHONPATH': GGUF_PY + os.pathsep + os.environ.get('PYTHONPATH', '')}
    subprocess.run([sys.executable, str(reference), 'pack', str(source), str(packed)], check=True, env=env)
    subprocess.run([sys.executable, str(reference), 'verify', str(packed), '--source', str(source)], check=True, env=env)
    command = [str(args.binary.resolve()), '-b', args.backend, '--kpack-load', str(packed.resolve())]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(name, argv, environment, expected_rc, marker):
        log = args.output / (name + '.log')
        with log.open('x') as out:
            result = subprocess.run(argv, env=environment, stdout=out, stderr=subprocess.STDOUT)
        text = log.read_text(errors='replace')
        accepted_rc = result.returncode != 0 if expected_rc is None else result.returncode == expected_rc
        if not accepted_rc or marker not in text:
            raise RuntimeError(f'{name}: rc={result.returncode}; see {log}')
        print(f'KPACK_LOAD_CHECK case={name} status=PASS log={log}', flush=True)
        return text

    text = run('online-offline', command, env, 0, 'KPACK_LOAD PASS tensors=10 inference=DISABLED')
    if 'KPACK_LOAD_DEFAULT native=PASS automatic_kpack=DISABLED' not in text:
        raise RuntimeError('default buffer isolation was not checked')
    run('forward-rejected', command + ['--kpack-reject-forward'], env, None,
        'K-pack is load-only; inference is not enabled')
    run('missing-packer', command, env | {'QUACTLIZE_PPU_PACK_LIBRARY': str(args.output / 'missing.so')}, 1,
        'KPACK_LOAD FAIL: packer is unavailable')
    print(f'KPACK_LOAD_SUITE PASS formats=5 tensors=10 results={args.output}', flush=True)


if __name__ == '__main__':
    main()
