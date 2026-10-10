// Runtime K-pack cache. Payload bytes are trusted; only metadata is validated.
#include "llama-kpack-sidecar.h"

#include "ggml.h"

// Keep file-format tests independent of libllama and device backends.
extern "C" void ggml_log_internal(enum ggml_log_level level, const char * format, ...);
#define GGML_LOG_INFO(...) ggml_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

using json  = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace {

constexpr int64_t      KPACK_ALIGN        = 128;
constexpr const char * KPACK_CACHE_SCHEMA = "llama.kpack-cache";

inline uint64_t align_up(uint64_t v) {
    return (v + KPACK_ALIGN - 1) & ~(uint64_t) (KPACK_ALIGN - 1);
}

// The k-quant names as the GGUF Python package spells them (GGMLQuantizationType.name): what the schema stores.
const char * kquant_type_name(int32_t t) {
    switch (t) {
        case GGML_TYPE_Q8_0:
            return "Q8_0";
        case GGML_TYPE_Q2_K:
            return "Q2_K";
        case GGML_TYPE_Q3_K:
            return "Q3_K";
        case GGML_TYPE_Q4_K:
            return "Q4_K";
        case GGML_TYPE_Q5_K:
            return "Q5_K";
        case GGML_TYPE_Q6_K:
            return "Q6_K";
        default:
            return nullptr;
    }
}

// Names and IDs from the existing runtime cache schema.
const char * layout_name_for(int32_t t) {
    return t == GGML_TYPE_Q8_0 ? "q8-kpack2" : t == GGML_TYPE_Q4_K ? "q4-kpack4" : "kquant-kpack";
}

int32_t layout_id_for(int32_t t) {
    return t == GGML_TYPE_Q8_0 ? 4 : t == GGML_TYPE_Q4_K ? 1 : 2;
}

// One mapping of a regular file opened without following a symlink, with the stability check the schema asks for.
struct mapped_file {
    int       fd   = -1;
    uint8_t * ptr  = nullptr;
    size_t    size = 0;

    struct stat st_before {};

    bool open(const std::string & path, bool nofollow, std::string & error, bool map_data = true) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (nofollow ? O_NOFOLLOW : 0));
        if (fd < 0) {
            error = path + ": " + strerror(errno);
            return false;
        }
        if (fstat(fd, &st_before) != 0 || !S_ISREG(st_before.st_mode)) {
            error = path + ": not a regular file";
            ::close(fd);
            fd = -1;
            return false;
        }
        size = (size_t) st_before.st_size;
        if (size && map_data) {
            void * m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) {
                error = path + ": mmap: " + strerror(errno);
                ::close(fd);
                fd = -1;
                return false;
            }
            ptr = (uint8_t *) m;
        }
        return true;
    }

    bool still_same(std::string & error) const {
        struct stat st {};

        if (fstat(fd, &st) != 0) {
            error = "fstat failed";
            return false;
        }
        if (st.st_dev != st_before.st_dev || st.st_ino != st_before.st_ino || st.st_size != st_before.st_size ||
            st.st_mtim.tv_sec != st_before.st_mtim.tv_sec || st.st_mtim.tv_nsec != st_before.st_mtim.tv_nsec ||
            st.st_ctim.tv_sec != st_before.st_ctim.tv_sec || st.st_ctim.tv_nsec != st_before.st_ctim.tv_nsec) {
            error = "file changed while it was being read";
            return false;
        }
        return true;
    }

    void close() {
        if (ptr) {
            munmap(ptr, size);
            ptr = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    ~mapped_file() { close(); }
};

json source_identity(const struct stat & st) {
    return {{"device", st.st_dev},
            {"inode", st.st_ino},
            {"mtime_sec", st.st_mtim.tv_sec},
            {"mtime_nsec", st.st_mtim.tv_nsec},
            {"ctime_sec", st.st_ctim.tv_sec},
            {"ctime_nsec", st.st_ctim.tv_nsec}};
}

// Strict JSON: the schema forbids duplicate keys, which nlohmann silently collapses. Track keys per open object.
bool parse_strict(const std::string & text, json & out, std::string & error) {
    std::vector<std::set<std::string>> scopes;
    bool                               dup = false;
    std::string                        dup_key;
    auto                               cb = [&](int /*depth*/, json::parse_event_t ev, json & parsed) -> bool {
        if (ev == json::parse_event_t::object_start) {
            scopes.emplace_back();
        } else if (ev == json::parse_event_t::object_end) {
            if (!scopes.empty()) {
                scopes.pop_back();
            }
        } else if (ev == json::parse_event_t::key) {
            const std::string k = parsed.get<std::string>();
            if (!scopes.empty() && !scopes.back().insert(k).second) {
                dup     = true;
                dup_key = k;
            }
        }
        return true;
    };
    try {
        out = json::parse(text, cb, true);
    } catch (const std::exception & e) {
        error = std::string("manifest.json: ") + e.what();
        return false;
    }
    if (dup) {
        error = "manifest.json: duplicate key '" + dup_key + "'";
        return false;
    }
    return true;
}

bool keys_exactly(const json & o, std::initializer_list<const char *> keys, const char * what, std::string & error) {
    if (!o.is_object()) {
        error = std::string(what) + " must be an object";
        return false;
    }
    std::set<std::string> want(keys.begin(), keys.end()), got;
    for (auto it = o.begin(); it != o.end(); ++it) {
        got.insert(it.key());
    }
    if (got != want) {
        error = std::string(what) + " must contain exactly {";
        for (auto k : keys) {
            error += k;
            error += ",";
        }
        error += "}";
        return false;
    }
    return true;
}

bool get_nonneg(const json & v, int64_t & out, const char * what, std::string & error) {
    if (!v.is_number_integer() || v.is_boolean()) {
        error = std::string(what) + " must be an integer";
        return false;
    }
    if (v.is_number_unsigned() && v.get<uint64_t>() > INT64_MAX) {
        error = std::string(what) + " is too large";
        return false;
    }
    const int64_t x = v.get<int64_t>();
    if (x < 0) {
        error = std::string(what) + " must be nonnegative";
        return false;
    }
    out = x;
    return true;
}

bool get_positive(const json & v, int64_t & out, const char * what, std::string & error) {
    if (!get_nonneg(v, out, what, error)) {
        return false;
    }
    if (out == 0) {
        error = std::string(what) + " must be positive";
        return false;
    }
    return true;
}

bool get_string(const json & v, std::string & out, const char * what, std::string & error) {
    if (!v.is_string()) {
        error = std::string(what) + " must be a string";
        return false;
    }
    out = v.get<std::string>();
    if (out.empty()) {
        error = std::string(what) + " must be nonempty";
        return false;
    }
    return true;
}

// Packed units pair superblocks when their metadata is not a multiple of four bytes.
struct geometry {
    int64_t              e1        = 1;    // experts, or 1 for dense
    int64_t              low_bytes = 0, high_bytes = 0, units_bytes = 0;
    int64_t              spu = 1, ub = 0;  // superblocks per packed unit, unit bytes
    std::vector<int64_t> low_shape, high_shape, units_shape;
};

bool derive_geometry(int64_t       n,
                     int64_t       k,
                     int64_t       experts,
                     bool          grouped,
                     int32_t       bits,
                     int32_t       high_bits,
                     int64_t       units_bytes_total,
                     geometry &    g,
                     std::string & error) {
    g.e1 = grouped ? experts : 1;
    if (n <= 0 || k <= 0 || g.e1 <= 0 || bits <= 0 || bits > 8 || high_bits < 0 || high_bits > 2 || n > INT32_MAX ||
        k > INT32_MAX || g.e1 > INT32_MAX || n > INT64_MAX / k / g.e1 / 256 || k % 256) {
        error = "invalid geometry";
        return false;
    }
    g.low_bytes   = g.e1 * n * k * bits / 8;
    g.high_bytes  = high_bits ? g.e1 * n * k * high_bits / 8 : 0;
    g.units_bytes = units_bytes_total;
    if (bits == 8) {
        if (high_bits || units_bytes_total != g.e1 * n * k / 16) {
            error = "Q8_0 requires only one FP16 scale per 32 codes";
            return false;
        }
        g.spu         = 1;
        g.ub          = 16;
        g.low_shape   = {g.e1, k / 2, n, 2};
        g.high_shape  = {0};
        g.units_shape = grouped ? std::vector<int64_t>{g.e1, k / 32, n, 2} : std::vector<int64_t>{k / 32, n, 2};
        return true;
    }
    if (units_bytes_total <= 0 || units_bytes_total > INT64_MAX / 256 || units_bytes_total % g.e1) {
        error = "invalid units plane size";
        return false;
    }
    const int64_t units_e = units_bytes_total / g.e1;
    if ((units_e * 256) % (n * k)) {
        error = "units plane is not a whole number of bytes per superblock";
        return false;
    }
    const int64_t sb = units_e * 256 / (n * k);
    g.spu            = (sb % 4) ? 2 : 1;
    g.ub             = g.spu * sb;
    if (k % (256 * g.spu)) {
        error = "K is not a multiple of the packed-unit extent";
        return false;
    }
    g.low_shape   = {g.e1, n, k * bits / 8};
    g.high_shape  = high_bits ? std::vector<int64_t>{g.e1, n, k * high_bits / 8} : std::vector<int64_t>{0};
    g.units_shape = grouped ? std::vector<int64_t>{experts, k / (256 * g.spu), n, g.ub} :
                              std::vector<int64_t>{k / (256 * g.spu), n, g.ub};
    return true;
}

int64_t shape_product(const std::vector<int64_t> & s) {
    int64_t p = 1;
    for (auto x : s) {
        if (x && p > INT64_MAX / x) {
            return -1;
        }
        p *= x;
    }
    return p;
}

}  // namespace

std::string llama_kpack_cache_directory(const std::string & source, std::string & error) {
    error.clear();

    struct stat st {};

    if (stat(source.c_str(), &st) != 0) {
        error = source + ": " + strerror(errno);
        return {};
    }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) {
        error = source + ": not a nonempty regular file";
        return {};
    }
    const char * configured = getenv("LLAMA_KPACK_CACHE_DIR");
    const char * tmp        = getenv("TMPDIR");
    std::string  root       = configured && *configured ?
                                  configured :
                                  std::string(tmp && *tmp ? tmp : "/tmp") + "/llama-kpack-" + std::to_string(geteuid());
    while (root.size() > 1 && root.back() == '/') {
        root.pop_back();
    }
    if (mkdir(root.c_str(), 0700) != 0 && errno != EEXIST) {
        error = root + ": " + strerror(errno);
        return {};
    }

    struct stat cache {};

    if (lstat(root.c_str(), &cache) != 0) {
        error = root + ": " + strerror(errno);
        return {};
    }
    // Cached bytes are trusted. Do not accept a shared or substituted directory.
    if (!S_ISDIR(cache.st_mode) || cache.st_uid != geteuid() || (cache.st_mode & 0022)) {
        error = root + ": cache root must be an owned directory, not writable by other users";
        return {};
    }
    std::string key = "v1";
    for (uint64_t value : {uint64_t(st.st_dev), uint64_t(st.st_ino), uint64_t(st.st_size), uint64_t(st.st_mtim.tv_sec),
                           uint64_t(st.st_mtim.tv_nsec), uint64_t(st.st_ctim.tv_sec), uint64_t(st.st_ctim.tv_nsec)}) {
        key += "-" + std::to_string(value);
    }
    return root + "/" + key;
}

struct llama_kpack_sidecar_reader::impl {
    json        manifest;
    mapped_file weights;
    uint64_t    source_size  = 0;
    uint64_t    storage_size = 0;
};

llama_kpack_sidecar_reader::llama_kpack_sidecar_reader() : pimpl(new impl) {}

llama_kpack_sidecar_reader::~llama_kpack_sidecar_reader() = default;

const llama_kpack_sidecar_record * llama_kpack_sidecar_reader::find(const std::string & name) const {
    auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : &records[it->second];
}

bool llama_kpack_sidecar_reader::open(const std::string & dir, std::string & error) try {
    pimpl.reset(new impl);
    records.clear();
    by_name.clear();
    root = dir;

    struct stat st {};

    if (lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        error = "K-pack sidecar root must be a real directory: " + dir;
        return false;
    }
    // Exactly two entries, nothing else -- an unlisted file is a bundle that did not finish or was tampered with.
    {
        std::set<std::string> entries;
        DIR *                 d = opendir(dir.c_str());
        if (!d) {
            error = "cannot list " + dir;
            return false;
        }
        while (dirent * e = readdir(d)) {
            const std::string nm = e->d_name;
            if (nm != "." && nm != "..") {
                entries.insert(nm);
            }
        }
        closedir(d);
        if (entries != std::set<std::string>{"manifest.json", "weights.bin"}) {
            error = "K-pack sidecar root entries disagree with the schema (want exactly manifest.json and weights.bin)";
            return false;
        }
    }

    std::string text;
    {
        mapped_file mf;
        if (!mf.open(dir + "/manifest.json", true, error)) {
            return false;
        }
        if (mf.size) {
            text.assign((const char *) mf.ptr, mf.size);
        }
        if (!mf.still_same(error)) {
            return false;
        }
    }
    json & m = pimpl->manifest;
    if (text.empty()) {
        error = "empty manifest";
        return false;
    }
    if (!parse_strict(text, m, error)) {
        return false;
    }

    if (!keys_exactly(m,
                      {"schema", "schema_version", "arrangement_version", "model", "selection", "source", "storage",
                       "tensors", "skipped"},
                      "K-pack manifest", error)) {
        return false;
    }
    if (m["schema"] != KPACK_CACHE_SCHEMA || (m["schema_version"] != 1 && m["schema_version"] != 2)) {
        error = "unsupported K-pack cache schema/version";
        return false;
    }
    if (m["arrangement_version"] != 2) {
        error = "production K-pack bundles require placed arrangement version 2";
        return false;
    }
    std::string model;
    if (!get_string(m["model"], model, "model", error)) {
        return false;
    }

    const json & src = m["source"];
    if (!keys_exactly(src, {"format", "size_bytes", "identity"}, "source", error)) {
        return false;
    }
    if (src["format"] != "gguf") {
        error = "source.format must be gguf";
        return false;
    }
    int64_t tmp;
    if (!get_positive(src["size_bytes"], tmp, "source.size_bytes", error)) {
        return false;
    }
    pimpl->source_size = (uint64_t) tmp;
    if (!keys_exactly(src["identity"], {"device", "inode", "mtime_sec", "mtime_nsec", "ctime_sec", "ctime_nsec"},
                      "source.identity", error)) {
        return false;
    }

    const json & sto = m["storage"];
    if (!keys_exactly(sto, {"file", "size_bytes", "alignment_bytes"}, "storage", error)) {
        return false;
    }
    if (sto["file"] != "weights.bin") {
        error = "storage.file must be weights.bin";
        return false;
    }
    if (!get_positive(sto["size_bytes"], tmp, "storage.size_bytes", error)) {
        return false;
    }
    pimpl->storage_size = (uint64_t) tmp;
    if (sto["alignment_bytes"] != KPACK_ALIGN) {
        error = "storage alignment must be 128";
        return false;
    }

    const json & sel = m["selection"];
    if (!keys_exactly(sel, {"layout_policy", "packable_total", "packed", "skipped"}, "selection", error)) {
        return false;
    }
    if (sel["layout_policy"] != "production-kpack-only") {
        error = "layout_policy must be production-kpack-only";
        return false;
    }
    int64_t packable_total, packed, nskipped;
    if (!get_nonneg(sel["packable_total"], packable_total, "selection.packable_total", error) ||
        !get_nonneg(sel["packed"], packed, "selection.packed", error) ||
        !get_nonneg(sel["skipped"], nskipped, "selection.skipped", error)) {
        return false;
    }
    if (!m["tensors"].is_array() || m["tensors"].empty()) {
        error = "a production K-pack bundle must contain at least one tensor";
        return false;
    }
    if (!m["skipped"].is_array()) {
        error = "skipped must be a list";
        return false;
    }
    if (packed != (int64_t) m["tensors"].size()) {
        error = "selection.packed disagrees with tensors length";
        return false;
    }
    if (nskipped != (int64_t) m["skipped"].size()) {
        error = "selection.skipped disagrees with skipped length";
        return false;
    }
    if (packable_total != packed) {
        error = "selection.packable_total must equal packed for a complete bundle";
        return false;
    }

    std::set<std::string> skipped_names;
    for (const auto & r : m["skipped"]) {
        if (!keys_exactly(r, {"name", "type_name", "reason"}, "skipped record", error)) {
            return false;
        }
        std::string nm, tn, why;
        if (!get_string(r["name"], nm, "skipped.name", error) ||
            !get_string(r["type_name"], tn, "skipped.type_name", error) ||
            !get_string(r["reason"], why, "skipped.reason", error)) {
            return false;
        }
        if (!skipped_names.insert(nm).second) {
            error = "duplicate tensor name in skipped: " + nm;
            return false;
        }
    }

    // ---- records ----
    records.clear();
    by_name.clear();
    uint64_t expected_region = 0;
    int64_t  prev_index      = -1;
    uint64_t prev_end        = 0;
    for (size_t i = 0; i < m["tensors"].size(); ++i) {
        const json &               r = m["tensors"][i];
        llama_kpack_sidecar_record rec;
        const std::string          where = "tensor record " + std::to_string(i);
        if (!keys_exactly(r,
                          {"name", "ggml_type", "type_name", "route_class", "layout_name", "plane_packs", "rank", "n",
                           "k", "experts", "arrangement_version", "arrangement", "source_tensor", "region", "spans"},
                          where.c_str(), error)) {
            return false;
        }
        if (!get_string(r["name"], rec.name, "name", error)) {
            return false;
        }
        const std::string pre = "artifact " + rec.name + ": ";
        int64_t           t;
        if (!get_nonneg(r["ggml_type"], t, "ggml_type", error)) {
            return false;
        }
        if (t > INT32_MAX) {
            error = "invalid ggml_type";
            return false;
        }
        rec.ggml_type      = (int32_t) t;
        const char * tname = kquant_type_name(rec.ggml_type);
        if (!tname) {
            error = pre + "not a production K-pack qtype";
            return false;
        }
        if (r["type_name"] != tname) {
            error = pre + "type_name disagrees with ggml_type";
            return false;
        }
        if (!get_string(r["route_class"], rec.route_class, "route_class", error)) {
            return false;
        }
        const bool grouped = rec.route_class == "grouped";
        if (!grouped && rec.route_class != "dense") {
            error = pre + "route_class must be dense or grouped";
            return false;
        }
        int64_t rank;
        if (!get_nonneg(r["rank"], rank, "rank", error)) {
            return false;
        }
        if ((grouped && rank != 3) || (!grouped && rank != 2)) {
            error = pre + "route_class/rank disagree";
            return false;
        }
        if (!get_positive(r["n"], rec.n, "n", error) || !get_positive(r["k"], rec.k, "k", error)) {
            return false;
        }
        if (grouped) {
            if (!get_positive(r["experts"], rec.experts, "experts", error)) {
                return false;
            }
        } else {
            if (!r["experts"].is_null()) {
                error = pre + "dense tensor must record experts=null";
                return false;
            }
            rec.experts = 0;
        }
        if (rec.n % 256 || rec.k % 256 ||
            ((rec.ggml_type == GGML_TYPE_Q3_K || rec.ggml_type == GGML_TYPE_Q6_K) && rec.k % 512)) {
            error = pre + "geometry outside the resident K-pack domain";
            return false;
        }
        if (r["layout_name"] != layout_name_for(rec.ggml_type)) {
            error = pre + "layout_name is not canonical";
            return false;
        }
        if (r["arrangement_version"] != 2) {
            error = pre + "requires arrangement_version=2";
            return false;
        }

        const json & a = r["arrangement"];
        if (!keys_exactly(a,
                          {"layout", "bits", "high_bits", "artifact_tile_k", "transport_tile_k", "group_size",
                           "reserved", "mapping_id"},
                          "arrangement", error)) {
            return false;
        }
        int64_t      f[8];
        const char * fn[8] = {"layout",           "bits",       "high_bits", "artifact_tile_k",
                              "transport_tile_k", "group_size", "reserved",  "mapping_id"};
        for (int j = 0; j < 8; ++j) {
            if (!get_nonneg(a[fn[j]], f[j], fn[j], error)) {
                error = pre + error;
                return false;
            }
        }
        for (int j = 0; j < 7; ++j) {
            if (f[j] > INT32_MAX) {
                error = pre + "arrangement field is too large";
                return false;
            }
        }
        auto & pl           = rec.planes;
        pl.layout           = (int32_t) f[0];
        pl.bits             = (int32_t) f[1];
        pl.high_bits        = (int32_t) f[2];
        pl.artifact_tile_k  = (int32_t) f[3];
        pl.transport_tile_k = (int32_t) f[4];
        pl.group_size       = (int32_t) f[5];
        pl.reserved         = (int32_t) f[6];
        pl.mapping_id       = (uint64_t) f[7];
        if (pl.layout != layout_id_for(rec.ggml_type) || pl.artifact_tile_k != 0 || pl.reserved != 0 || pl.bits <= 0 ||
            pl.bits > 8 || pl.high_bits < 0 || pl.high_bits > 2) {
            error = pre + "arrangement is not a canonical K-pack descriptor";
            return false;
        }
        const json & pp = r["plane_packs"];
        if (!keys_exactly(pp, {"low", "high"}, "plane_packs", error)) {
            return false;
        }
        if (pp["low"] != 16 / pl.bits || pp["high"] != (pl.high_bits ? 16 / pl.high_bits : 0)) {
            error = pre + "plane_packs disagree with the arrangement";
            return false;
        }

        if (rec.n > INT32_MAX || rec.k > INT32_MAX || rec.experts > INT32_MAX ||
            rec.n > INT64_MAX / rec.k / std::max<int64_t>(1, rec.experts) / 256) {
            error = pre + "geometry is too large";
            return false;
        }
        const int64_t raw_bytes =
            (int64_t) ggml_row_size((ggml_type) rec.ggml_type, rec.k) * rec.n * (grouped ? rec.experts : 1);

        const json & s = r["source_tensor"];
        if (!keys_exactly(s, {"index", "data_offset", "size_bytes"}, "source_tensor", error)) {
            return false;
        }
        int64_t sidx, soff, ssize;
        if (!get_nonneg(s["index"], sidx, "source_tensor.index", error) ||
            !get_nonneg(s["data_offset"], soff, "source_tensor.data_offset", error) ||
            !get_positive(s["size_bytes"], ssize, "source_tensor.size_bytes", error)) {
            return false;
        }
        if (ssize != raw_bytes) {
            error = pre + "source tensor size is not the canonical GGUF size";
            return false;
        }
        const json & rg = r["region"];
        if (!keys_exactly(rg, {"offset_bytes", "size_bytes"}, "region", error)) {
            return false;
        }
        int64_t roff, rsize;
        if (!get_nonneg(rg["offset_bytes"], roff, "region.offset_bytes", error) ||
            !get_positive(rg["size_bytes"], rsize, "region.size_bytes", error)) {
            return false;
        }
        if (roff % KPACK_ALIGN || rsize % KPACK_ALIGN) {
            error = pre + "region must be 128-byte aligned";
            return false;
        }
        rec.region_offset = (uint64_t) roff;
        rec.region_size   = (uint64_t) rsize;

        const json & sp = r["spans"];
        if (!keys_exactly(sp, {"low", "high", "units"}, "spans", error)) {
            return false;
        }
        llama_kpack_sidecar_record::span * spans[3]      = {&rec.low, &rec.high, &rec.units};
        const char *                       span_names[3] = {"low", "high", "units"};
        for (int j = 0; j < 3; ++j) {
            const json & v = sp[span_names[j]];
            if (!keys_exactly(v, {"offset_bytes", "size_bytes", "shape"}, "span", error)) {
                return false;
            }
            int64_t o, z;
            if (!get_nonneg(v["offset_bytes"], o, "span.offset_bytes", error) ||
                !get_nonneg(v["size_bytes"], z, "span.size_bytes", error)) {
                return false;
            }
            if (!v["shape"].is_array()) {
                error = pre + "span shape must be a list";
                return false;
            }
            for (const auto & x : v["shape"]) {
                int64_t d;
                if (!get_nonneg(x, d, "shape", error)) {
                    return false;
                }
                spans[j]->shape.push_back(d);
            }
            spans[j]->offset = (uint64_t) o;
            spans[j]->size   = (uint64_t) z;
        }
        geometry g;
        if (!derive_geometry(rec.n, rec.k, rec.experts, grouped, pl.bits, pl.high_bits, (int64_t) rec.units.size, g,
                             error)) {
            error = pre + error;
            return false;
        }
        const std::vector<int64_t> * want_shape[3] = {&g.low_shape, &g.high_shape, &g.units_shape};
        uint64_t                     cursor        = 0;
        for (int j = 0; j < 3; ++j) {
            if (spans[j]->offset != align_up(cursor)) {
                error = pre + std::string(span_names[j]) + " span offset is not canonical";
                return false;
            }
            if (spans[j]->shape != *want_shape[j]) {
                error = pre + std::string(span_names[j]) + " shape is not canonical";
                return false;
            }
            if ((int64_t) spans[j]->size != shape_product(spans[j]->shape)) {
                error = pre + std::string(span_names[j]) + " size is not canonical";
                return false;
            }
            cursor = spans[j]->offset + spans[j]->size;
        }
        if (cursor != (uint64_t) raw_bytes) {
            error = pre + "planes must be contiguous and byte-neutral";
            return false;
        }
        if ((int64_t) rec.low.size != g.low_bytes || (int64_t) rec.high.size != g.high_bytes) {
            error = pre + "plane sizes disagree with the arrangement";
            return false;
        }
        if (rec.region_size != align_up(cursor)) {
            error = pre + "region size does not cover its spans";
            return false;
        }
        if ((int64_t) rec.region_size != raw_bytes) {
            error = pre + "region is not byte-neutral with its GGUF tensor";
            return false;
        }

        // ordering across records
        if (by_name.count(rec.name)) {
            error = "duplicate tensor name in K-pack manifest: " + rec.name;
            return false;
        }
        if (rec.region_offset != expected_region) {
            error = pre + "region is not in canonical manifest order";
            return false;
        }
        if (sidx <= prev_index) {
            error = "K-pack source tensor indices must be strictly increasing";
            return false;
        }
        if ((uint64_t) soff < prev_end) {
            error = "K-pack source tensor byte ranges must be ordered and disjoint";
            return false;
        }
        if (rec.region_offset > pimpl->storage_size || rec.region_size > pimpl->storage_size - rec.region_offset ||
            (uint64_t) soff > pimpl->source_size || (uint64_t) ssize > pimpl->source_size - (uint64_t) soff) {
            error = pre + "region outside file";
            return false;
        }
        expected_region += rec.region_size;
        prev_index = sidx;
        prev_end   = (uint64_t) soff + (uint64_t) ssize;

        pl.low_bytes      = rec.low.size;
        pl.high_bytes     = rec.high.size;
        pl.units_bytes    = rec.units.size;
        by_name[rec.name] = records.size();
        records.push_back(std::move(rec));
    }
    for (const auto & rec : records) {
        if (skipped_names.count(rec.name)) {
            error = "K-pack tensors and skipped inventory overlap: " + rec.name;
            return false;
        }
    }
    if (expected_region != pimpl->storage_size) {
        error = "K-pack tensor regions do not cover storage.size_bytes exactly";
        return false;
    }

    if (!pimpl->weights.open(dir + "/weights.bin", true, error)) {
        return false;
    }
    if (pimpl->weights.size != pimpl->storage_size) {
        error = "K-pack storage size mismatch: expected " + std::to_string(pimpl->storage_size) + " observed " +
                std::to_string(pimpl->weights.size);
        return false;
    }
    return true;
}

catch (const std::exception & e) {
    records.clear();
    by_name.clear();
    error = e.what();
    return false;
}

bool llama_kpack_sidecar_reader::check_source_metadata(const std::vector<llama_kpack_source_tensor> & inventory,
                                                       std::string &                                  error) {
    std::map<std::string, const llama_kpack_source_tensor *> inv;
    for (const auto & t : inventory) {
        inv[t.name] = &t;
    }

    // Bind each record to the loader's tensor inventory, not just its name.
    for (const auto & rec : records) {
        auto it = inv.find(rec.name);
        if (it == inv.end()) {
            error = "sidecar tensor " + rec.name + " is not in the model";
            return false;
        }
        const auto & t       = *it->second;
        const json & s       = pimpl->manifest["tensors"][by_name.at(rec.name)]["source_tensor"];
        const bool   grouped = rec.route_class == "grouped";
        if (t.gguf_index != s["index"].get<int64_t>() || t.data_offset != s["data_offset"].get<uint64_t>() ||
            t.size_bytes != s["size_bytes"].get<uint64_t>() || t.ggml_type != rec.ggml_type ||
            t.rank != (grouped ? 3 : 2) || t.n != rec.n || t.k != rec.k || t.experts != (grouped ? rec.experts : 0)) {
            error = "sidecar tensor " + rec.name + ": identity (index/offset/size/type/shape) disagrees with the GGUF";
            return false;
        }
        if (t.data_offset > pimpl->source_size || t.size_bytes > pimpl->source_size - t.data_offset) {
            error = "sidecar tensor " + rec.name + " range is outside the GGUF";
            return false;
        }
    }
    return true;
}

bool llama_kpack_sidecar_reader::load_unchecked(const std::string &                            gguf_path,
                                                const std::vector<llama_kpack_source_tensor> & inventory,
                                                std::string &                                  error) {
    if (!check_source_metadata(inventory, error)) {
        return false;
    }
    mapped_file source;
    if (!source.open(gguf_path, false, error, false)) {
        return false;
    }
    if (source.size != pimpl->source_size ||
        source_identity(source.st_before) != pimpl->manifest["source"]["identity"]) {
        error = "source GGUF identity differs from the local cache";
        return false;
    }
    if (!pimpl->weights.still_same(error)) {
        return false;
    }
    resolve_planes();
    return true;
}

void llama_kpack_sidecar_reader::resolve_planes() {
    const auto & w = pimpl->weights;
    for (auto & rec : records) {
        const uint8_t * base = w.ptr + rec.region_offset;
        rec.planes.low       = base + rec.low.offset;
        rec.planes.high      = rec.high.size ? base + rec.high.offset : nullptr;
        rec.planes.units     = base + rec.units.offset;
    }
}

struct llama_kpack_sidecar_writer::impl {
    std::string           final_dir, staging;
    int                   fd         = -1;
    uint64_t              pos        = 0;
    ojson                 tensors    = ojson::array();
    ojson                 skipped    = ojson::array();
    int64_t               prev_index = -1;
    uint64_t              prev_end   = 0;
    std::set<std::string> names;
    mapped_file           source;
    std::string           source_path;

    bool write_all(const void * data, size_t size, std::string & error) {
        const uint8_t * p = (const uint8_t *) data;
        while (size) {
            const ssize_t n = ::write(fd, p, size);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                error = std::string("weights.bin: write: ") + (n == 0 ? "no progress" : strerror(errno));
                return false;
            }
            p += n;
            size -= (size_t) n;
            pos += (uint64_t) n;
        }
        return true;
    }

    bool pad_to(uint64_t target, std::string & error) {
        static const uint8_t zeros[4096] = {0};
        while (pos < target) {
            if (!write_all(zeros, (size_t) std::min<uint64_t>(sizeof(zeros), target - pos), error)) {
                return false;
            }
        }
        return true;
    }

    void remove_staging() {
        if (staging.empty()) {
            return;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        ::unlink((staging + "/weights.bin").c_str());
        ::unlink((staging + "/manifest.json").c_str());
        ::rmdir(staging.c_str());
        staging.clear();
    }
};

llama_kpack_sidecar_writer::llama_kpack_sidecar_writer() : pimpl(new impl) {}

llama_kpack_sidecar_writer::~llama_kpack_sidecar_writer() {
    abort();
}

size_t llama_kpack_sidecar_writer::packed() const {
    return pimpl->tensors.size();
}

void llama_kpack_sidecar_writer::abort() {
    pimpl->remove_staging();
    pimpl->source.close();
}

bool llama_kpack_sidecar_writer::bind_source(const std::string & path, std::string & error, int loader_fd) {
    if (pimpl->source.fd >= 0) {
        error = "source is already bound";
        return false;
    }
    if (!pimpl->source.open(path, false, error, false)) {
        return false;
    }
    if (loader_fd >= 0) {
        struct stat st {};

        const auto & bound = pimpl->source.st_before;
        if (fstat(loader_fd, &st) != 0 || st.st_dev != bound.st_dev || st.st_ino != bound.st_ino ||
            st.st_size != bound.st_size || st.st_mtim.tv_sec != bound.st_mtim.tv_sec ||
            st.st_mtim.tv_nsec != bound.st_mtim.tv_nsec || st.st_ctim.tv_sec != bound.st_ctim.tv_sec ||
            st.st_ctim.tv_nsec != bound.st_ctim.tv_nsec) {
            error = "source path is not the loader's file";
            pimpl->source.close();
            return false;
        }
    }
    pimpl->source_path = path;
    return true;
}

bool llama_kpack_sidecar_writer::begin(const std::string & dir, std::string & error) {
    if (pimpl->fd >= 0 || !pimpl->staging.empty()) {
        error = "writer is already open";
        return false;
    }

    struct stat st {};

    if (lstat(dir.c_str(), &st) == 0) {
        error = "refusing to overwrite existing output " + dir;
        return false;
    }
    if (errno != ENOENT) {
        error = dir + ": lstat: " + strerror(errno);
        return false;
    }
    const std::string staging = dir + ".partial." + std::to_string((long) getpid());
    if (mkdir(staging.c_str(), 0755) != 0) {
        error = staging + ": mkdir: " + strerror(errno);
        return false;
    }
    // Cleanup may only own a directory this writer created successfully.
    pimpl->final_dir = dir;
    pimpl->staging   = staging;
    pimpl->fd        = ::open((pimpl->staging + "/weights.bin").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (pimpl->fd < 0) {
        error = std::string("weights.bin: ") + strerror(errno);
        pimpl->remove_staging();
        return false;
    }
    return true;
}

void llama_kpack_sidecar_writer::skip(const std::string & name,
                                      const std::string & type_name,
                                      const std::string & reason) {
    pimpl->skipped.push_back(ojson{{"name", name}, {"type_name", type_name}, {"reason", reason}});
}

bool llama_kpack_sidecar_writer::add_stream(const llama_kpack_source_tensor & src,
                                            const llama_kpack_planes &        planes,
                                            const read_chunk &                read,
                                            size_t                            chunk_bytes,
                                            const cancelled &                 cancel,
                                            std::string &                     error) {
    const auto & source = pimpl->source;
    if (source.fd < 0 || src.data_offset > source.size || src.size_bytes > source.size - src.data_offset) {
        error = "source range is outside the bound file";
        return false;
    }
    return add_record(src, planes, read, chunk_bytes, cancel, error);
}

bool llama_kpack_sidecar_writer::add_record(const llama_kpack_source_tensor & src,
                                            const llama_kpack_planes &        planes,
                                            const read_chunk &                read,
                                            size_t                            chunk_bytes,
                                            const cancelled &                 cancel,
                                            std::string &                     error) {
    auto &            I   = *pimpl;
    const std::string pre = src.name + ": ";
    if (I.fd < 0) {
        error = "writer is not open";
        return false;
    }
    if (!read || chunk_bytes == 0) {
        error = pre + "invalid chunk reader";
        return false;
    }
    const char * tname = kquant_type_name(src.ggml_type);
    if (!tname) {
        error = pre + "not a K-pack format";
        return false;
    }
    const bool grouped = src.rank == 3;
    if (!(src.rank == 2 || src.rank == 3) || (grouped && src.experts <= 0) || (!grouped && src.experts != 0)) {
        error = pre + "rank/experts disagree";
        return false;
    }
    if (src.gguf_index <= I.prev_index) {
        error = pre + "records must be added in increasing GGUF index order";
        return false;
    }
    if (src.data_offset < I.prev_end) {
        error = pre + "source byte ranges must be ascending and disjoint";
        return false;
    }
    if (!I.names.insert(src.name).second) {
        error = pre + "duplicate tensor";
        return false;
    }

    geometry g;
    if (!derive_geometry(src.n, src.k, src.experts, grouped, planes.bits, planes.high_bits,
                         (int64_t) planes.units_bytes, g, error)) {
        error = pre + error;
        return false;
    }
    if ((int64_t) planes.low_bytes != g.low_bytes || (int64_t) planes.high_bytes != g.high_bytes) {
        error = pre + "plane sizes disagree with the arrangement";
        return false;
    }
    const uint64_t raw_bytes = ggml_row_size((ggml_type) src.ggml_type, src.k) * (uint64_t) src.n * (uint64_t) g.e1;
    if (src.size_bytes != raw_bytes) {
        error = pre + "source size is not the canonical GGUF size";
        return false;
    }

    // region
    const uint64_t region_offset = align_up(I.pos);
    if (!I.pad_to(region_offset, error)) {
        return false;
    }
    const uint64_t               sizes[3]        = {planes.low_bytes, planes.high_bytes, planes.units_bytes};
    const std::vector<int64_t> * shapes[3]       = {&g.low_shape, &g.high_shape, &g.units_shape};
    const char *                 span_names[3]   = {"low", "high", "units"};
    ojson                        spans           = ojson::object();
    uint64_t                     cursor          = 0;
    size_t                       resident_offset = 0;
    for (int j = 0; j < 3; ++j) {
        const uint64_t off = align_up(cursor);
        if (!I.pad_to(region_offset + off, error)) {
            return false;
        }
        for (size_t copied = 0; copied < sizes[j];) {
            if (cancel && cancel()) {
                error = "cache write cancelled";
                return false;
            }
            const size_t    count = std::min<size_t>(chunk_bytes, sizes[j] - copied);
            const uint8_t * data  = read(resident_offset + copied, count, error);
            if (!data) {
                return false;
            }
            if (!I.write_all(data, count, error)) {
                return false;
            }
            copied += count;
        }
        ojson shape = ojson::array();
        for (auto d : *shapes[j]) {
            shape.push_back(d);
        }
        spans[span_names[j]] = ojson{{"offset_bytes", off}, {"size_bytes", sizes[j]}, {"shape", shape}};
        resident_offset += sizes[j];
        cursor = off + sizes[j];
    }
    const uint64_t region_size = align_up(cursor);
    if (!I.pad_to(region_offset + region_size, error)) {
        return false;
    }
    if (region_size != raw_bytes) {
        error = pre + "resident region is not byte-neutral with the GGUF tensor";
        return false;
    }

    ojson arr = ojson{
        {"layout", planes.layout},
        {"bits", planes.bits},
        {"high_bits", planes.high_bits},
        {"artifact_tile_k", planes.artifact_tile_k},
        {"transport_tile_k", planes.transport_tile_k},
        {"group_size", planes.group_size},
        {"reserved", planes.reserved},
        {"mapping_id", planes.mapping_id},
    };
    ojson rec = ojson{
        {"name", src.name},
        {"ggml_type", src.ggml_type},
        {"type_name", tname},
        {"route_class", grouped ? "grouped" : "dense"},
        {"layout_name", layout_name_for(src.ggml_type)},
        {"plane_packs", ojson{{"low", 16 / planes.bits}, {"high", planes.high_bits ? 16 / planes.high_bits : 0}}},
        {"rank", src.rank},
        {"n", src.n},
        {"k", src.k},
        {"experts", grouped ? ojson(src.experts) : ojson(nullptr)},
        {"arrangement_version", 2},
        {"arrangement", arr},
        {"source_tensor",
         ojson{{"index", src.gguf_index}, {"data_offset", src.data_offset}, {"size_bytes", src.size_bytes}}},
        {"region", ojson{{"offset_bytes", region_offset}, {"size_bytes", region_size}}},
        {"spans", spans},
    };
    I.tensors.push_back(std::move(rec));
    I.prev_index = src.gguf_index;
    I.prev_end   = src.data_offset + src.size_bytes;
    return true;
}

static bool fsync_path(const std::string & path, std::string & error) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = path + ": " + strerror(errno);
        return false;
    }
    const bool ok = fsync(fd) == 0;
    if (!ok) {
        error = path + ": fsync: " + strerror(errno);
    }
    ::close(fd);
    return ok;
}

static bool publish_cache_files(const std::string & staging, const std::string & target, std::string & error) {
    const int source_fd = open(staging.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (source_fd < 0) {
        error = staging + ": open: " + strerror(errno);
        return false;
    }
    if (mkdir(target.c_str(), 0755) != 0) {
        error = target + ": exclusive mkdir: " + strerror(errno);
        close(source_fd);
        return false;
    }
    const int target_fd = open(target.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (target_fd < 0) {
        error = target + ": open: " + strerror(errno);
        close(source_fd);
        return false;
    }
    bool weights_linked = false;
    bool published      = false;
    bool ok             = false;
    // Readers require the manifest. Publish it only after the data link is durable.
    if (linkat(source_fd, "weights.bin", target_fd, "weights.bin", 0) != 0) {
        error = target + ": link weights.bin: " + strerror(errno);
    } else {
        weights_linked = true;
        if (fsync(target_fd) != 0) {
            error = target + ": sync data link: " + strerror(errno);
        } else if (linkat(source_fd, "manifest.json", target_fd, "manifest.json", 0) != 0) {
            error = target + ": publish manifest.json: " + strerror(errno);
        } else {
            published = true;
            ok        = fsync(target_fd) == 0;
            if (!ok) {
                error = target + ": sync published manifest: " + strerror(errno);
            }
        }
    }
    if (!published) {
        // Remove only this writer's link, never a replacement made by another writer.
        struct stat source {
        }, linked{}, owned{}, current{};

        if (weights_linked && fstatat(source_fd, "weights.bin", &source, AT_SYMLINK_NOFOLLOW) == 0 &&
            fstatat(target_fd, "weights.bin", &linked, AT_SYMLINK_NOFOLLOW) == 0 && source.st_dev == linked.st_dev &&
            source.st_ino == linked.st_ino) {
            unlinkat(target_fd, "weights.bin", 0);
        }
        if (fstat(target_fd, &owned) == 0 && lstat(target.c_str(), &current) == 0 && owned.st_dev == current.st_dev &&
            owned.st_ino == current.st_ino) {
            rmdir(target.c_str());
        }
    }
    close(target_fd);
    close(source_fd);
    return ok;
}

bool llama_kpack_sidecar_writer::finish(const std::string & model_label,
                                        const std::string & gguf_path,
                                        std::string &       error,
                                        const cancelled &   cancel) {
    auto & I = *pimpl;
    if (I.fd < 0) {
        error = "writer is not open";
        return false;
    }
    if (I.tensors.empty()) {
        error = "refusing to create an empty artifact bundle";
        abort();
        return false;
    }
    if (fsync(I.fd) != 0) {
        error = std::string("weights.bin: fsync: ") + strerror(errno);
        abort();
        return false;
    }
    ::close(I.fd);
    I.fd = -1;

    uint64_t src_size = 0;
    json     identity;
    {
        mapped_file temporary;
        if (I.source.fd < 0 && !temporary.open(gguf_path, false, error, false)) {
            abort();
            return false;
        }
        const mapped_file & mf = I.source.fd >= 0 ? I.source : temporary;

        struct stat current {};

        if (stat(gguf_path.c_str(), &current) != 0 || current.st_dev != mf.st_before.st_dev ||
            current.st_ino != mf.st_before.st_ino || (I.source.fd >= 0 && gguf_path != I.source_path)) {
            error = "source path changed since load";
            abort();
            return false;
        }
        src_size = mf.size;
        identity = source_identity(mf.st_before);
        if (!mf.still_same(error)) {
            abort();
            return false;
        }
        for (const auto & r : I.tensors) {
            const auto &   s   = r["source_tensor"];
            const uint64_t off = s["data_offset"].get<uint64_t>(), bytes = s["size_bytes"].get<uint64_t>();
            if (off > src_size || bytes > src_size - off) {
                error = "source range is outside file";
                abort();
                return false;
            }
        }
    }
    if (src_size == 0) {
        error = "source GGUF must not be empty";
        abort();
        return false;
    }

    ojson manifest = ojson{
        {"schema", KPACK_CACHE_SCHEMA},
        {"schema_version", 1},
        {"arrangement_version", 2},
        {"model", model_label},
        {"source", ojson{{"format", "gguf"}, {"size_bytes", src_size}, {"identity", identity}}},
        {"storage", ojson{{"file", "weights.bin"}, {"size_bytes", I.pos}, {"alignment_bytes", KPACK_ALIGN}}},
        {"selection", ojson{{"layout_policy", "production-kpack-only"},
                            {"packable_total", I.tensors.size()},
                            {"packed", I.tensors.size()},
                            {"skipped", I.skipped.size()}}},
        {"tensors", I.tensors},
        {"skipped", I.skipped},
    };
    const std::string text = manifest.dump(2) + "\n";
    const int mfd = ::open((I.staging + "/manifest.json").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (mfd < 0) {
        error = std::string("manifest.json: ") + strerror(errno);
        abort();
        return false;
    }
    {
        const char * p    = text.data();
        size_t       left = text.size();
        while (left) {
            const ssize_t n = ::write(mfd, p, left);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                error = std::string("manifest.json: write: ") + strerror(errno);
                ::close(mfd);
                abort();
                return false;
            }
            p += n;
            left -= (size_t) n;
        }
        if (fsync(mfd) != 0) {
            error = "manifest.json: fsync failed";
            ::close(mfd);
            abort();
            return false;
        }
        ::close(mfd);
    }
    if (!fsync_path(I.staging, error)) {
        abort();
        return false;
    }
    if (cancel && cancel()) {
        error = "cache write cancelled";
        abort();
        return false;
    }

    // Keep no-replace semantics on filesystems without renameat2 flags.
#ifndef RENAME_NOREPLACE
#    define RENAME_NOREPLACE (1 << 0)
#endif
    if (syscall(SYS_renameat2, AT_FDCWD, I.staging.c_str(), AT_FDCWD, I.final_dir.c_str(), RENAME_NOREPLACE) != 0) {
        const int saved_errno = errno;
        if (saved_errno != EINVAL && saved_errno != ENOSYS && saved_errno != EOPNOTSUPP) {
            error = I.final_dir + ": renameat2(RENAME_NOREPLACE): " + strerror(saved_errno);
            abort();
            return false;
        }
        if (!publish_cache_files(I.staging, I.final_dir, error)) {
            abort();
            return false;
        }
        I.remove_staging();
        GGML_LOG_INFO("[kpack-cache] published with manifest-last links: %s\n", I.final_dir.c_str());
    } else {
        I.staging.clear();
    }
    I.source.close();

    std::string  parent = I.final_dir;
    const size_t slash  = parent.find_last_of('/');
    parent              = slash == std::string::npos ? "." : (slash == 0 ? "/" : parent.substr(0, slash));
    if (!fsync_path(parent, error)) {
        return false;
    }
    error.clear();
    GGML_LOG_INFO("[kpack] wrote %zu artifact(s) to %s\n", I.tensors.size(), I.final_dir.c_str());
    return true;
}

struct llama_kpack_background_writer::impl {
    llama_kpack_sidecar_writer writer;
    std::string                source_path, error;
    std::thread                worker;
    std::atomic<bool>          cancelled{false};
    bool                       prepared = false, started = false, success = false;
};

llama_kpack_background_writer::llama_kpack_background_writer() : pimpl(new impl) {}

llama_kpack_background_writer::~llama_kpack_background_writer() {
    cancel();
}

bool llama_kpack_background_writer::prepare(const std::string & dir,
                                            const std::string & source_path,
                                            std::string &       error,
                                            int                 loader_fd) {
    auto & I = *pimpl;
    if (I.prepared || I.started) {
        error = "background writer is already prepared";
        return false;
    }
    if (!I.writer.bind_source(source_path, error, loader_fd) || !I.writer.begin(dir, error)) {
        I.writer.abort();
        return false;
    }
    I.source_path = source_path;
    I.prepared    = true;
    return true;
}

bool llama_kpack_background_writer::start(std::vector<llama_kpack_write_job>             jobs,
                                          const std::vector<llama_kpack_source_tensor> & inventory,
                                          size_t                                         chunk_bytes,
                                          std::string &                                  error) {
    auto & I = *pimpl;
    if (!I.prepared || I.started || I.cancelled || jobs.empty() || chunk_bytes == 0) {
        error = "background writer cannot start";
        return false;
    }
    std::sort(jobs.begin(), jobs.end(), [](const llama_kpack_write_job & a, const llama_kpack_write_job & b) {
        return a.source.gguf_index < b.source.gguf_index;
    });
    std::set<std::string> names;
    for (const auto & job : jobs) {
        if (!job.read || !names.insert(job.source.name).second) {
            error = "invalid or duplicate snapshot";
            return false;
        }
    }
    for (const auto & src : inventory) {
        if (!names.count(src.name)) {
            I.writer.skip(src.name, ggml_type_name((ggml_type) src.ggml_type), "not resident in a K-pack buffer");
        }
    }
    try {
        I.worker = std::thread([&I, jobs = std::move(jobs), chunk_bytes]() {
            const auto cancel = [&I]() {
                return I.cancelled.load();
            };
            const auto began  = std::chrono::steady_clock::now();
            auto       last   = began;
            size_t     copied = 0, completed = 0;
            try {
                for (const auto & job : jobs) {
                    if (!I.writer.add_stream(job.source, job.planes, job.read, chunk_bytes, cancel, I.error)) {
                        I.writer.abort();
                        return;
                    }
                    copied += job.planes.low_bytes + job.planes.high_bytes + job.planes.units_bytes;
                    ++completed;
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last >= std::chrono::seconds(5) || completed == jobs.size()) {
                        GGML_LOG_INFO(
                            "[kpack-cache] write progress: tensors=%zu/%zu MiB=%.1f seconds=%.2f "
                            "content_checks=disabled\n",
                            completed, jobs.size(), copied / (1024.0 * 1024.0),
                            std::chrono::duration<double>(now - began).count());
                        last = now;
                    }
                }
                const auto flushing = std::chrono::steady_clock::now();
                GGML_LOG_INFO("[kpack-cache] flushing cache to disk\n");
                I.success = I.writer.finish(I.source_path, I.source_path, I.error, cancel);
                if (I.success) {
                    const auto now = std::chrono::steady_clock::now();
                    GGML_LOG_INFO("[kpack-cache] published: total_seconds=%.2f flush_seconds=%.2f\n",
                                  std::chrono::duration<double>(now - began).count(),
                                  std::chrono::duration<double>(now - flushing).count());
                }
            } catch (const std::exception & e) {
                I.error = e.what();
            } catch (...) {
                I.error = "unknown background cache failure";
            }
            if (!I.success) {
                I.writer.abort();
            }
        });
    } catch (const std::exception & e) {
        error = e.what();
        I.writer.abort();
        return false;
    }
    I.started = true;
    return true;
}

bool llama_kpack_background_writer::wait(std::string & error) {
    auto & I = *pimpl;
    if (I.worker.joinable()) {
        I.worker.join();
    }
    error = I.error;
    return I.started && I.success;
}

void llama_kpack_background_writer::cancel() {
    pimpl->cancelled = true;
    if (pimpl->worker.joinable()) {
        pimpl->worker.join();
    }
    pimpl->writer.abort();
}
