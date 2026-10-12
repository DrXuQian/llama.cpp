#!/usr/bin/env python3
"""Compare the actual buffer's GPU shuffle with offline reference GGUF bytes."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

GGUF_PY = str(Path(__file__).resolve().parents[1] / 'gguf-py')
sys.path.insert(0, GGUF_PY)
import gguf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        help='offline gguf_kpack.py; defaults to quactlize/gguf_kpack.py beside --binary')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cache-binary', type=Path, help='also test background persistence and cross-process hits')
    parser.add_argument('--packer', type=Path, default=os.environ.get('QUACTLIZE_PPU_PACK_LIBRARY'),
                        help='GPU pack library; defaults to QUACTLIZE_PPU_PACK_LIBRARY or the library beside --binary')
    parser.add_argument('--output', type=Path,
                        help='new directory; default: temporary kpack-load-* directory, removed on success')
    parser.add_argument('--backend', default='CUDA0')
    parser.add_argument('--tp-devices', default=os.environ.get('KPACK_TP_DEVICES'),
                        help='comma-separated TP devices; default: two logical ranks on --backend')
    args = parser.parse_args()
    if args.packer is None:
        args.packer = args.binary.resolve().parent / 'libquactlize_ppu_pack.so'
    if args.reference is None:
        args.reference = args.binary.resolve().parent / 'quactlize' / 'gguf_kpack.py'
    for path in (args.binary, args.packer, args.reference):
        if not path.is_file():
            parser.error(f'missing {path}')
    if args.cache_binary is not None and not args.cache_binary.is_file():
        parser.error(f'missing {args.cache_binary}')
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


def q8_fixture(path):
    """Independent byte oracle: pair k/k+8, transpose N, preserve scale bits."""
    writer = gguf.GGUFWriter(path, 'llama')
    records = []
    for n, k, experts in ((256, 512, 1), (256, 512, 3), (8448, 1024, 1)):
        name = f'q8.n{n}.e{experts}.weight'
        raw = np.random.default_rng(n + experts).integers(0, 256, (experts, n, k // 32, 34), dtype=np.uint8)
        codes = raw[..., 2:] ^ np.uint8(128)
        low = codes.reshape(experts, n, k // 16, 2, 8).transpose(0, 2, 4, 1, 3).copy()
        units = raw[..., :2].transpose(0, 2, 1, 3).copy()
        recovered = np.empty_like(raw)
        recovered[..., 2:] = low.transpose(0, 3, 1, 4, 2).reshape(codes.shape) ^ np.uint8(128)
        recovered[..., :2] = units.transpose(0, 2, 1, 3)
        if not np.array_equal(raw, recovered):
            raise RuntimeError('Q8 byte oracle inverse differs')
        writer.add_tensor(name, raw.reshape(experts, n, -1), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        for plane, data in (('low', low), ('units', units)):
            writer.add_tensor(name + '.' + plane, data.reshape(-1).view(np.int8))
        records.append({
            'source_name': name, 'qtype': 8, 'n': n, 'k': k, 'experts': experts,
            'arrangement': {'version': 2, 'layout': 4, 'bits': 8, 'high_bits': 0,
                            'artifact_tile_k': 0, 'transport_tile_k': 32, 'group_size': 32,
                            'reserved': 0, 'mapping_id': 0x51384b5032540001},
            'carriers': {'low': {'name': name + '.low'}, 'high': None, 'units': {'name': name + '.units'}},
        })
    writer.add_string('quactlize.kpack.reference.manifest', json.dumps({
        'schema': 'quactlize.kquant-kpack.reference-gguf', 'schema_version': 1, 'tensors': records,
    }))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def tp_fixture(path, reference, count):
    import torch
    spec = importlib.util.spec_from_file_location('kpack_reference', reference)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    writer = gguf.GGUFWriter(path, 'llama')
    records = []
    for q in (8, 10, 11, 12, 13, 14):
        block, block_bytes = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(q)]
        for case, axis in (('k', 0), ('n', 1), ('experts', 2), ('dim3', 3), ('mirror', 10),
                           ('uneven-empty', 1), ('segments', 0)):
            ne = [512, 256, 1, 1]
            widths, repeats = [], []
            coordinates = [[] for _ in range(count)]
            if axis < 4:
                unit = 512 if axis == 0 else 256 if axis == 1 else 1
                widths = [unit] * count
                repeats = [1]
                if case == 'uneven-empty':
                    widths = [unit * (d + 1) for d in range(count - 1)] + [0]
                if case == 'segments':
                    widths = [256] * count + [512] * count
                    repeats = [2, 1]
                position = 0
                for s, repeat in enumerate(repeats):
                    for _ in range(repeat):
                        for d in range(count):
                            width = widths[s * count + d]
                            coordinates[d].extend(range(position, position + width))
                            position += width
                ne[axis] = position
            if axis == 3:
                ne[2] = 2
            name = f'tp.q{q}.{case}.weight'
            raw = np.random.default_rng(q * 100 + axis).integers(
                0, 256, (ne[3], ne[2], ne[1], ne[0] // block, block_bytes), dtype=np.uint8)
            writer.add_tensor(name, raw.reshape(ne[3], ne[2], ne[1], -1), raw_dtype=gguf.GGMLQuantizationType(q))
            shards = []
            for d in range(count):
                if axis < 4 and not coordinates[d]:
                    shards.append({'carriers': {}})
                    continue
                indices = coordinates[d]
                if axis == 0:
                    indices = np.asarray(indices[::block]) // block
                local = raw if axis == 10 else np.take(raw, indices, axis=3-axis)
                local = np.ascontiguousarray(local)
                e = local.shape[0] * local.shape[1]
                n, k = local.shape[2], local.shape[3] * block
                if q == 8:
                    blocks = local.reshape(e, n, k // 32, 34)
                    codes = blocks[..., 2:] ^ np.uint8(128)
                    low = codes.reshape(e, n, k // 16, 2, 8).transpose(0, 2, 4, 1, 3).copy()
                    high = np.empty(0, dtype=np.uint8)
                    units = blocks[..., :2].transpose(0, 2, 1, 3).copy()
                else:
                    art = module.prepare_grouped(torch.from_numpy(local.reshape(-1, block_bytes)), n, k, q, e)
                    low, high, units = (plane.numpy() for plane in (art.low, art.high, art.units))
                    recovered = module.recover_raw_blocks(art).numpy()
                    if not np.array_equal(recovered.reshape(-1), local.reshape(-1)):
                        raise RuntimeError(f'TP reference inverse differs: {name}/{d}')
                carriers = {}
                for plane, data in (('low', low), ('high', high), ('units', units)):
                    carrier = f'{name}.rank{d}.{plane}'
                    carriers[plane] = {'name': carrier} if data.size else None
                    if data.size:
                        writer.add_tensor(carrier, data.reshape(-1).view(np.int8))
                shards.append({'carriers': carriers})
            records.append({'source_name': name, 'partition': {'axis': axis, 'widths': widths, 'repeats': repeats},
                            'shards': shards})
            print(f'KPACK_TP_FIXTURE q={q} case={case} ranks={count}', flush=True)
    writer.add_string('quactlize.kpack.reference.manifest', json.dumps({'tensors': records}))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


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
    command = [str(args.binary.resolve()), '-b', args.backend, str(packed.resolve())]
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
    run('forward-rejected', command + ['--reject-forward'], env, None,
        'K-pack is load-only; inference is not enabled')
    run('missing-packer', command, env | {'QUACTLIZE_PPU_PACK_LIBRARY': str(args.output / 'missing.so')}, 1,
        'KPACK_LOAD FAIL: packer is unavailable')
    if args.cache_binary:
        q8 = args.output / 'q8-reference.gguf'
        q8_fixture(q8)
        run('q8-online-offline', [str(args.binary.resolve()), '-b', args.backend, str(q8)], env, 0,
            'KPACK_LOAD PASS tensors=3 inference=DISABLED')
        for fixture, label, tensors in ((packed, 'kquant', 10), (q8, 'q8', 3)):
            cache = args.output / ('cache-' + label)
            cached = [str(args.cache_binary.resolve()), args.backend, str(fixture.resolve()), str(cache)]
            run(label + '-cache-cold', cached + ['cold'], env, 0,
                f'KPACK_CACHE PASS mode=cold tensors={tensors} hits=0 gpu_shuffle={tensors}')
            manifest_path = cache / 'manifest.json'
            original = manifest_path.read_bytes()
            run(label + '-cache-hot', cached + ['hot'], env, 0,
                f'KPACK_CACHE PASS mode=hot tensors={tensors} hits={tensors} gpu_shuffle=0')
            run(label + '-cache-forward-rejected', cached + ['hot', 'reject'], env, None,
                'K-pack is load-only; inference is not enabled')
            changed = json.loads(original)
            for tensor in changed['tensors']:
                tensor['arrangement']['mapping_id'] ^= 1
            manifest_path.write_text(json.dumps(changed))
            run(label + '-cache-layout-miss', cached + ['incompatible'], env, 0,
                f'KPACK_CACHE PASS mode=incompatible tensors={tensors} hits=0 gpu_shuffle={tensors}')
            # Keep the bad cache intact; a miss must not overwrite an existing target.
            changed = json.loads(original)
            changed['source']['identity']['inode'] += 1
            manifest_path.write_text(json.dumps(changed))
            rejected = manifest_path.read_bytes()
            run(label + '-cache-stale', cached + ['miss'], env, 0,
                f'KPACK_CACHE PASS mode=miss tensors={tensors} hits=0 gpu_shuffle={tensors}')
            if manifest_path.read_bytes() != rejected:
                raise RuntimeError('cache miss replaced an existing directory')
            partial = cached[:-1] + [str(args.output / ('cache-partial-' + label))]
            run(label + '-cache-partial-cold', partial + ['cold-partial'], env, 0,
                f'KPACK_CACHE PASS mode=cold-partial tensors={tensors} hits=0 gpu_shuffle={tensors}')
            run(label + '-cache-partial-hot', partial + ['hot-partial'], env, 0,
                f'KPACK_CACHE PASS mode=hot-partial tensors={tensors} hits=1 gpu_shuffle={tensors - 1}')
        print(f'KPACK_CACHE_SUITE PASS formats=6 tensors=13 results={args.output}', flush=True)
        devices = args.tp_devices or f'{args.backend},{args.backend}'
        count = len(devices.split(','))
        if not 2 <= count <= 16:
            raise ValueError('--tp-devices needs 2..16 logical ranks')
        tp = args.output / 'tp-reference.gguf'
        tp_fixture(tp, args.reference, count)
        command = [str(args.cache_binary.resolve()), devices, str(tp), str(args.output / 'cache-tp')]
        run('tp-cache-cold', command + ['cold', 'tp'], env, 0, f'KPACK_TP_CACHE PASS ranks={count}')
        run('tp-cache-hot', command + ['hot', 'tp'], env, 0, 'hits=42 gpu_shuffle=0')
        run('tp-cache-forward-rejected', command + ['hot', 'tp-reject'], env, None,
            'K-pack is load-only; inference is not enabled')
        manifest_path = args.output / 'cache-tp' / 'manifest.json'
        changed = json.loads(manifest_path.read_text())
        for record in changed['tensors']:
            record['arrangement']['mapping_id'] ^= 1
        manifest_path.write_text(json.dumps(changed))
        run('tp-cache-layout-miss', command + ['incompatible', 'tp'], env, 0, 'hits=0 gpu_shuffle=42')
        partial = command[:-1] + [str(args.output / 'cache-tp-partial')]
        run('tp-cache-partial-cold', partial + ['cold-partial', 'tp'], env, 0, 'hits=0 gpu_shuffle=42')
        run('tp-cache-partial-hot', partial + ['hot-partial', 'tp'], env, 0, 'hits=1 gpu_shuffle=41')
        print(f'KPACK_TP_SUITE PASS formats=6 cases=42 logical_ranks={count} devices={devices}', flush=True)
    print(f'KPACK_LOAD_SUITE PASS formats=5 tensors=10 results={args.output}', flush=True)


if __name__ == '__main__':
    main()
