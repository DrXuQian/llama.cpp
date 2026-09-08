#pragma once

// Persistent K-pack sidecar: quactlize's "kquant-kpack.bundle" schema v3, read and written from llama.cpp.
//
// WHAT IT IS. One directory next to the model -- manifest.json + weights.bin -- holding, for every k-quant tensor
// that lives in the K-pack buffer type, the three resident planes (low / high / units) exactly as the kernels read
// them, plus enough identity to prove they were built from THIS GGUF: the source file's size and whole-file SHA-256,
// each source tensor's byte range and digest, and each plane's digest. The format, every field, every check and the
// alignment rule are quactlize's (quactlize/pack_gguf.py is the authority and its load_kpack_bundle is the oracle a
// bundle written here has to pass); nothing about the layout is decided in this file.
//
// The GPU producer creates the planes once. Later loads verify and upload the
// persisted bytes without repeating conversion; source identity is not an mtime-only check.
//
// TWO ROLES, ONE FORMAT. The reader validates a bundle and hands planes to the buffer type; the writer captures the
// planes set_tensor just produced and publishes a bundle atomically (never over an existing one). Both are host-only
// and know nothing about devices; llama-model-loader wires them to the K-pack buffer type through the backend's
// proc-address seams.

#include "ggml.h"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// The planes of one tensor, as bytes in memory. Mirrors ggml_quactlize_planes without depending on the backend
// header; the loader copies fields across.
struct llama_kpack_planes {
    const uint8_t * low   = nullptr; size_t low_bytes   = 0;
    const uint8_t * high  = nullptr; size_t high_bytes  = 0;   // null / 0 when the format has no high plane
    const uint8_t * units = nullptr; size_t units_bytes = 0;
    // quactlize_ppu_placed_arrangement_v2, field for field, in declaration order. Kept as plain integers so this
    // header stays free of the backend ABI header; the loader converts.
    int32_t  layout = 0, bits = 0, high_bits = 0, artifact_tile_k = 0, transport_tile_k = 0, group_size = 0, reserved = 0;
    uint64_t mapping_id = 0;
};

// What the loader knows about one source tensor, used to bind a bundle record to the GGUF it came from.
struct llama_kpack_source_tensor {
    std::string name;
    int32_t     gguf_index  = -1;    // position in the GGUF tensor table
    uint64_t    data_offset = 0;     // absolute byte offset of its data in the file
    uint64_t    size_bytes  = 0;
    int32_t     ggml_type   = -1;
    int32_t     rank        = 0;
    int64_t     n = 0, k = 0, experts = 0;   // GGUF [K, N(, E)] -> route terms; experts = 0 for dense
};

// ---- reader ----

struct llama_kpack_sidecar_record {
    std::string name;
    int32_t     ggml_type = -1;
    std::string route_class;            // "dense" | "grouped"
    int64_t     n = 0, k = 0, experts = 0;
    uint64_t    region_offset = 0, region_size = 0;
    struct span { uint64_t offset = 0, size = 0; std::vector<int64_t> shape; std::string sha256; } low, high, units;
    llama_kpack_planes planes;          // pointers resolve into the mapped weights.bin once verified
};

class llama_kpack_sidecar_reader {
public:
    llama_kpack_sidecar_reader();
    ~llama_kpack_sidecar_reader();

    // Parse and structurally validate manifest.json (exact field sets, canonical shapes, ordered contiguous
    // regions, binding digests) and map weights.bin. Nothing is trusted yet: call verify_source and verify_storage.
    bool open(const std::string & dir, std::string & error);

    // Prove the bundle is about THIS GGUF: size and whole-file SHA-256 match manifest.source, every recorded
    // tensor exists in `inventory` with the same index / offset / size / type / shape, and every source byte range
    // hashes to its recorded digest. gguf_path is read once, sequentially; the ranges are hashed in parallel.
    bool verify_source(const std::string & gguf_path, const std::vector<llama_kpack_source_tensor> & inventory,
                       int n_threads, std::string & error);

    // Prove weights.bin is what the manifest says: size, whole-file SHA-256, every span's SHA-256, zero padding,
    // no unlisted tail. After this the records' plane pointers are valid.
    bool verify_storage(int n_threads, std::string & error);

    const llama_kpack_sidecar_record * find(const std::string & name) const;
    size_t size() const { return records.size(); }
    const std::string & dir() const { return root; }

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
    std::string root;
    std::vector<llama_kpack_sidecar_record> records;
    std::map<std::string, size_t> by_name;
};

// ---- writer ----

class llama_kpack_sidecar_writer {
public:
    llama_kpack_sidecar_writer();
    ~llama_kpack_sidecar_writer();

    // Create the staging directory `<dir>.partial.<pid>` and open its weights.bin. `dir` itself must not exist.
    bool begin(const std::string & dir, std::string & error);

    // Capture file identity, not loader-owned data. Source hashes are filled at
    // publication, separately from copying the packed planes.
    bool bind_source(const std::string & path, std::string & error, int loader_fd = -1);

    // Append one tensor's planes as the next region. Records MUST arrive in increasing gguf_index order with
    // disjoint, ascending byte ranges. The background writer sorts its jobs;
    // synchronous callers must supply this order themselves. source_data is hashed here.
    bool add(const llama_kpack_source_tensor & src, const void * source_data, const llama_kpack_planes & planes,
             std::string & error);

    using read_chunk = std::function<const uint8_t *(size_t offset, size_t bytes, std::string & error)>;
    using cancelled = std::function<bool()>;

    // Worker-only streaming append. Offsets address contiguous [low][high][units],
    // not the padded file region. The returned span lives until the next read.
    // No full-tensor host allocation or raw GGUF access in this operation.
    bool add_stream(const llama_kpack_source_tensor & src, const llama_kpack_planes & planes,
                    const read_chunk & read, size_t chunk_bytes, const cancelled & cancel, std::string & error);

    // Record a tensor that was NOT packed and why. The schema wants the full inventory of what was left out.
    void skip(const std::string & name, const std::string & type_name, const std::string & reason);

    // Write manifest.json, fsync everything, and publish the staging directory to `dir` with a no-replace rename.
    // model_label is what the manifest calls the model (the GGUF path); gguf_path is hashed whole for
    // manifest.source. On any failure the staging directory is removed and nothing is published.
    bool finish(const std::string & model_label, const std::string & gguf_path, std::string & error,
                const cancelled & cancel = {});

    // Abandon: remove the staging directory.
    void abort();

    size_t packed() const;

private:
    bool add_record(const llama_kpack_source_tensor & src, const std::string & source_sha,
                    const llama_kpack_planes & planes, const read_chunk & read, size_t chunk_bytes,
                    const cancelled & cancel, std::string & error);
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_kpack_write_job {
    llama_kpack_source_tensor source;
    llama_kpack_planes planes;
    llama_kpack_sidecar_writer::read_chunk read;
};

// One bounded streaming writer per model. start() only transfers metadata and
// starts the worker. Device weights and read callback resources must outlive wait()
// or cancel(). Neither operation belongs in the compute submission path.
class llama_kpack_background_writer {
public:
    llama_kpack_background_writer();
    ~llama_kpack_background_writer();
    bool prepare(const std::string & dir, const std::string & source_path, std::string & error, int loader_fd = -1);
    bool start(std::vector<llama_kpack_write_job> jobs,
               const std::vector<llama_kpack_source_tensor> & inventory,
               size_t chunk_bytes, std::string & error);
    bool wait(std::string & error);
    void cancel();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// SHA-256 of a byte range, lowercase hex. Exposed for tests and for the loader's own checks.
std::string llama_kpack_sha256_hex(const void * data, size_t size);

// Test hook: run the portable SHA-256 even where the CPU has the SHA extension, so the two can be compared.
void llama_kpack_sha256_force_portable(bool on);
