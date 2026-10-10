#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// LLAMA_KPACK_CACHE_DIR overrides ${TMPDIR:-/tmp}/llama-kpack-<uid>.
// Each source stat identity has its own subdirectory. Empty means unavailable.
std::string llama_kpack_cache_directory(const std::string & source, std::string & error);

// llama.kpack-cache v1: trusted local bytes, stat identity and tensor metadata.
// There are no payload hashes. Keep the source and cache immutable during use.
struct llama_kpack_planes {
    const uint8_t * low         = nullptr;
    size_t          low_bytes   = 0;
    const uint8_t * high        = nullptr;
    size_t          high_bytes  = 0;
    const uint8_t * units       = nullptr;
    size_t          units_bytes = 0;
    int32_t         layout = 0, bits = 0, high_bits = 0, artifact_tile_k = 0, transport_tile_k = 0, group_size = 0,
            reserved    = 0;
    uint64_t mapping_id = 0;
};

struct llama_kpack_source_tensor {
    std::string name;
    int32_t     gguf_index  = -1;
    uint64_t    data_offset = 0, size_bytes = 0;
    int32_t     ggml_type = -1, rank = 0;
    int64_t     n = 0, k = 0, experts = 0;
};

struct llama_kpack_sidecar_record {
    std::string name;
    int32_t     ggml_type = -1;
    std::string route_class;
    int64_t     n = 0, k = 0, experts = 0;
    uint64_t    region_offset = 0, region_size = 0;

    struct span {
        uint64_t             offset = 0, size = 0;
        std::vector<int64_t> shape;
    } low, high, units;

    llama_kpack_planes planes;
};

class llama_kpack_sidecar_reader {
public:
    llama_kpack_sidecar_reader();
    ~llama_kpack_sidecar_reader();
    bool                               open(const std::string & dir, std::string & error);
    bool                               load_unchecked(const std::string &                            source,
                                                      const std::vector<llama_kpack_source_tensor> & inventory,
                                                      std::string &                                  error);
    const llama_kpack_sidecar_record * find(const std::string & name) const;

    size_t size() const { return records.size(); }

private:
    bool check_source_metadata(const std::vector<llama_kpack_source_tensor> & inventory, std::string & error);
    void resolve_planes();
    struct impl;
    std::unique_ptr<impl>                   pimpl;
    std::string                             root;
    std::vector<llama_kpack_sidecar_record> records;
    std::map<std::string, size_t>           by_name;
};

class llama_kpack_sidecar_writer {
public:
    llama_kpack_sidecar_writer();
    ~llama_kpack_sidecar_writer();
    bool begin(const std::string & dir, std::string & error);
    bool bind_source(const std::string & path, std::string & error, int loader_fd = -1);

    using read_chunk = std::function<const uint8_t *(size_t offset, size_t bytes, std::string & error)>;
    using cancelled  = std::function<bool()>;

    // Worker-only. Read contiguous [low][high][units] in bounded, plane-limited chunks.
    bool   add_stream(const llama_kpack_source_tensor & src,
                      const llama_kpack_planes &        planes,
                      const read_chunk &                read,
                      size_t                            chunk_bytes,
                      const cancelled &                 cancel,
                      std::string &                     error);
    void   skip(const std::string & name, const std::string & type_name, const std::string & reason);
    bool   finish(const std::string & model_label,
                  const std::string & source,
                  std::string &       error,
                  const cancelled &   cancel = {});
    void   abort();
    size_t packed() const;

private:
    bool add_record(const llama_kpack_source_tensor & src,
                    const llama_kpack_planes &        planes,
                    const read_chunk &                read,
                    size_t                            chunk_bytes,
                    const cancelled &                 cancel,
                    std::string &                     error);
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_kpack_write_job {
    llama_kpack_source_tensor              source;
    llama_kpack_planes                     planes;
    llama_kpack_sidecar_writer::read_chunk read;
};

// Weights and read callbacks must outlive wait()/cancel(). Neither belongs in inference.
class llama_kpack_background_writer {
public:
    llama_kpack_background_writer();
    ~llama_kpack_background_writer();
    bool prepare(const std::string & dir, const std::string & source, std::string & error, int loader_fd = -1);
    bool start(std::vector<llama_kpack_write_job>             jobs,
               const std::vector<llama_kpack_source_tensor> & inventory,
               size_t                                         chunk_bytes,
               std::string &                                  error);
    bool wait(std::string & error);
    void cancel();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
