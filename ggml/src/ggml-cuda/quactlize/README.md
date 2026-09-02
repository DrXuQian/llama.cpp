# quactlize ABI headers (verbatim copies)

Copied unmodified from `quactlize/quactlize/include/` at source commit **`2826cf12451e02ca4590f7a44682b57d2098bfb9`**
-- the commit the published six-library bundle (`prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1`) was built from.
The handoff pins headers and libraries to each other: do not refresh these from another revision, even when a struct
or export name looks unchanged, without moving the bundle too. llama.cpp compiles against these and dlopen's the prebuilt
`libquactlize_ppu_fmt*.so`; it never builds quactlize sources. `ABI_SHA256` records the exact bytes this tree was
built against, so a drifted ABI is a one-line `sha256sum -c` away rather than a link error nobody reads:

    sha256sum -c ABI_SHA256

`ppu_format_config.inc` is quactlize's per-format registry, whose own header says shipping code must consume it or
make divergence fail loudly. It is used here ONLY as a cross-check on the arrangement descriptor the library hands
back -- llama.cpp does not construct an arrangement from it, because that would be a second source of the same
policy and the two could disagree silently.
