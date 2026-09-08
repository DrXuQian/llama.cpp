# quactlize ABI headers (verbatim copies)

Copied unmodified from `quactlize/quactlize/include/` at source commit **`2826cf12451e02ca4590f7a44682b57d2098bfb9`**
-- the commit the published six-library bundle (`prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1`) was built from.
The handoff pins headers and libraries to each other: do not refresh these from another revision, even when a struct
or export name looks unchanged, without moving the bundle too. llama.cpp compiles against these and dlopen's the prebuilt
`libquactlize_ppu_fmt*.so`; it never builds quactlize sources. `ABI_SHA256` records the exact bytes this tree was
built against, so a drifted ABI is a one-line `sha256sum -c` away rather than a link error nobody reads:

    sha256sum -c ABI_SHA256

`quactlize_ppu_pack.h` is an additional verbatim copy of
`quactlize/packing/api.h` at `7fbe892e2d774049437e9aa771f421cf633474ab`.
It describes the separate, five-format GPU producer `libquactlize_ppu_pack.so`;
it does not change the six-library consumer ABI above. Set
`QUACTLIZE_PPU_PACK_LIBRARY` to its absolute path, or place it in
`QUACTLIZE_PPU_BUNDLE`. A missing/incompatible GPU producer declines K-pack
intake rather than falling back to CPU conversion. No build manifest is read
by this runtime binding.

`ppu_format_config.inc` is quactlize's per-format registry, whose own header says shipping code must consume it or
make divergence fail loudly. It is used here ONLY as a cross-check on the arrangement descriptor the library hands
back -- llama.cpp does not construct an arrangement from it, because that would be a second source of the same
policy and the two could disagree silently.

## Runtime weight cache

`--kpack-cache DIR` persists GPU-produced resident planes. On a miss, a single
background writer uses two 8 MiB pinned slots per device. It prefetches the
next D2H range while writing the completed range, then fsyncs and atomically
publishes the directory. No completion wait is added to compute submission.
Model teardown still waits before releasing weights; copy and disk traffic
can contend for bandwidth with inference.

Runtime caches use `llama.kpack-cache` v1, with the existing K-pack plane
layout but **no content checksums**. Hits check file sizes, tensor metadata,
arrangements and bounds, then upload the mapped bytes directly. A local cache
also checks source device/inode/mtime/ctime, so copying or modifying the source
file may invalidate it. Payload corruption is not detected: use trusted cache
storage. An invalid/existing directory is not overwritten; use a new path to
repack. Source files must remain unchanged during use.

Old `quactlize.kquant-kpack.bundle` v3 files remain readable on this unchecked
runtime path. The explicit offline verifier still requires their checksums;
the new hash-free cache is not a verified v3 interchange bundle. No Quactlize
producer or GEMM library rebuild is needed for this host-side change.

`tests/run-quactlize-cache-smoke.sh` measures no-cache/cold/hit wall and model
timings. Optionally set `EVAL_FILE` to a corpus of at least 512 tokens for the
existing `llama-perplexity` PPL/KLD comparison: uncached K-pack vs cached K-pack,
and the ordinary GPU weight route vs K-pack. `EVAL_BATCH=128` exercises prefill;
`EVAL_BATCH=1` exercises teacher-forced decode. These are separate runs. The
tool's saved log probabilities are compressed, not a bitwise FP32 logit oracle.
Metrics require review; successful generation or exit 0 is not numerical
admission. Log-probability files remain on the box; result archives contain
only logs and the cache manifest.
