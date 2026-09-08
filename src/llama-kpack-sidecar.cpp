// Persistent K-pack sidecar (quactlize kquant-kpack.bundle schema v3). See llama-kpack-sidecar.h.
//
// Logging goes through ggml's log seam rather than llama's so this file depends on ggml-base only: the host-only
// test compiles it directly and never has to load a backend to exercise a file format.

#include "llama-kpack-sidecar.h"

#include "ggml.h"

// ggml's log seam, declared here rather than through ggml-impl.h: that header lives in ggml/src, off libllama's
// include path, and the only thing wanted from it is this one exported function.
extern "C" void ggml_log_internal(enum ggml_log_level level, const char * format, ...);
#define GGML_LOG_INFO(...) ggml_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
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

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#include <immintrin.h>
#define LLAMA_KPACK_HAVE_SHANI 1
#endif

using json  = nlohmann::json;
using ojson = nlohmann::ordered_json;

// ============================================================================================================
// SHA-256
// ============================================================================================================

namespace {

const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_blocks_portable(uint32_t st[8], const uint8_t * p, size_t nblocks) {
    uint32_t w[64];
    while (nblocks--) {
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i+1] << 16 | (uint32_t) p[4*i+2] << 8 | p[4*i+3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K256[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
        p += 64;
    }
}

#ifdef LLAMA_KPACK_HAVE_SHANI
// The SHA extension path: the widely used schedule (Intel's reference), written as one loop over the sixteen
// four-round groups instead of unrolled. Verified against the portable path by the test on random lengths --
// that comparison is the only reason to trust this function.
__attribute__((target("sha,sse4.1,ssse3")))
void sha256_blocks_shani(uint32_t st[8], const uint8_t * p, size_t nblocks) {
    const __m128i MASK = _mm_set_epi64x((long long) 0x0c0d0e0f08090a0bULL, (long long) 0x0405060700010203ULL);

    __m128i TMP    = _mm_loadu_si128((const __m128i *) &st[0]);
    __m128i STATE1 = _mm_loadu_si128((const __m128i *) &st[4]);
    TMP    = _mm_shuffle_epi32(TMP, 0xB1);
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);
    __m128i STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);

    while (nblocks--) {
        const __m128i ABEF_SAVE = STATE0;
        const __m128i CDGH_SAVE = STATE1;
        __m128i W[4];
        for (int g = 0; g < 16; ++g) {
            const int ci = g & 3, pi = (g + 3) & 3, ni = (g + 1) & 3;
            if (g < 4) {
                W[ci] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) (p + 16*g)), MASK);
            }
            __m128i MSG = _mm_add_epi32(W[ci], _mm_loadu_si128((const __m128i *) &K256[4*g]));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
            if (g >= 3 && g <= 14) {
                const __m128i T = _mm_alignr_epi8(W[ci], W[pi], 4);
                W[ni] = _mm_add_epi32(W[ni], T);
                W[ni] = _mm_sha256msg2_epu32(W[ni], W[ci]);
            }
            MSG = _mm_shuffle_epi32(MSG, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
            if (g >= 1 && g <= 14) {
                W[pi] = _mm_sha256msg1_epu32(W[pi], W[ci]);
            }
        }
        STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
        STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);
        p += 64;
    }

    TMP    = _mm_shuffle_epi32(STATE0, 0x1B);
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);
    _mm_storeu_si128((__m128i *) &st[0], STATE0);
    _mm_storeu_si128((__m128i *) &st[4], STATE1);
}

bool cpu_has_shani() {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) { return false; }
    const bool ssse3 = c & (1u << 9), sse41 = c & (1u << 19);
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) { return false; }
    const bool sha = b & (1u << 29);
    return ssse3 && sse41 && sha;
}
#endif

std::atomic<int> g_force_portable{0};

struct sha256_ctx {
    uint32_t st[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t  buf[64];
    size_t   buf_len = 0;
    uint64_t total   = 0;
    bool     shani   = false;

    sha256_ctx() {
#ifdef LLAMA_KPACK_HAVE_SHANI
        static const bool has = cpu_has_shani();
        shani = has && g_force_portable.load() == 0;
#endif
    }
    void blocks(const uint8_t * p, size_t n) {
#ifdef LLAMA_KPACK_HAVE_SHANI
        if (shani) { sha256_blocks_shani(st, p, n); return; }
#endif
        sha256_blocks_portable(st, p, n);
    }
    void update(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        total += size;
        if (buf_len) {
            const size_t take = std::min(size, (size_t) 64 - buf_len);
            memcpy(buf + buf_len, p, take);
            buf_len += take; p += take; size -= take;
            if (buf_len == 64) { blocks(buf, 1); buf_len = 0; }
        }
        const size_t nb = size / 64;
        if (nb) { blocks(p, nb); p += nb * 64; size -= nb * 64; }
        if (size) { memcpy(buf, p, size); buf_len = size; }
    }
    void final(uint8_t out[32]) {
        const uint64_t bits = total * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (buf_len != 56) { update(&zero, 1); }
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) { len[i] = (uint8_t) (bits >> (56 - 8*i)); }
        update(len, 8);
        for (int i = 0; i < 8; ++i) {
            out[4*i] = (uint8_t) (st[i] >> 24); out[4*i+1] = (uint8_t) (st[i] >> 16);
            out[4*i+2] = (uint8_t) (st[i] >> 8); out[4*i+3] = (uint8_t) st[i];
        }
    }
    std::string hex() {
        uint8_t d[32]; final(d);
        static const char * h = "0123456789abcdef";
        std::string s(64, '0');
        for (int i = 0; i < 32; ++i) { s[2*i] = h[d[i] >> 4]; s[2*i+1] = h[d[i] & 15]; }
        return s;
    }
};

// ============================================================================================================
// helpers
// ============================================================================================================

constexpr int64_t     KPACK_ALIGN   = 128;
constexpr const char * KPACK_SCHEMA = "quactlize.kquant-kpack.bundle";
constexpr int         KPACK_VERSION = 3;

inline uint64_t align_up(uint64_t v) { return (v + KPACK_ALIGN - 1) & ~(uint64_t) (KPACK_ALIGN - 1); }

bool is_hex64(const std::string & s) {
    if (s.size() != 64) { return false; }
    for (char c : s) { if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) { return false; } }
    return true;
}

// The k-quant names as the GGUF Python package spells them (GGMLQuantizationType.name): what the schema stores.
const char * kquant_type_name(int32_t t) {
    switch (t) {
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        default: return nullptr;
    }
}

std::string upper_type_name(int32_t t) {
    std::string s = ggml_type_name((ggml_type) t);
    for (auto & c : s) { c = (char) toupper((unsigned char) c); }
    return s;
}

// The layout policy the schema records per format: Q4_K rides its own descriptor, every other k-quant the shared one.
const char * layout_name_for(int32_t t) { return t == GGML_TYPE_Q4_K ? "q4-kpack4" : "kquant-kpack"; }
int32_t      layout_id_for(int32_t t)   { return t == GGML_TYPE_Q4_K ? 1 : 2; }

// Python's json.dumps(separators=(",", ":")) for one string: ensure_ascii, the short escapes json.dumps uses, and
// \uXXXX (surrogate pairs above the BMP) for everything else. The binding digest is over this exact encoding.
std::string json_compact_string(const std::string & s) {
    std::string out = "\"";
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp; int len;
        if      (c < 0x80)           { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else                         { cp = c & 0x07; len = 4; }
        for (int j = 1; j < len && i + j < s.size(); ++j) { cp = (cp << 6) | ((unsigned char) s[i+j] & 0x3F); }
        i += len;
        char tmp[16];
        if (cp == '"')       { out += "\\\""; }
        else if (cp == '\\') { out += "\\\\"; }
        else if (cp == '\n') { out += "\\n"; }
        else if (cp == '\r') { out += "\\r"; }
        else if (cp == '\t') { out += "\\t"; }
        else if (cp == '\b') { out += "\\b"; }
        else if (cp == '\f') { out += "\\f"; }
        else if (cp < 0x20 || (cp >= 0x7F && cp <= 0xFFFF)) { snprintf(tmp, sizeof(tmp), "\\u%04x", cp); out += tmp; }
        else if (cp > 0xFFFF) {
            cp -= 0x10000;
            snprintf(tmp, sizeof(tmp), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)); out += tmp;
        }
        else { out += (char) cp; }
    }
    return out + "\"";
}

std::string binding_digest(const std::string & name, int32_t ggml_type, int32_t rank, int64_t n, int64_t k,
                           int64_t experts, int64_t index, uint64_t data_offset, uint64_t size_bytes,
                           const std::string & sha) {
    std::string payload = "[" + json_compact_string(name) + "," + std::to_string(ggml_type) + "," +
        std::to_string(rank) + "," + std::to_string(n) + "," + std::to_string(k) + "," +
        (experts ? std::to_string(experts) : std::string("null")) + "," + std::to_string(index) + "," +
        std::to_string(data_offset) + "," + std::to_string(size_bytes) + "," + json_compact_string(sha) + "]";
    sha256_ctx h; h.update(payload.data(), payload.size());
    return h.hex();
}

// One mapping of a regular file opened without following a symlink, with the stability check the schema asks for.
struct mapped_file {
    int      fd   = -1;
    uint8_t * ptr = nullptr;
    size_t   size = 0;
    struct stat st_before{};

    bool open(const std::string & path, bool nofollow, std::string & error, bool map_data = true) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (nofollow ? O_NOFOLLOW : 0));
        if (fd < 0) { error = path + ": " + strerror(errno); return false; }
        if (fstat(fd, &st_before) != 0 || !S_ISREG(st_before.st_mode)) {
            error = path + ": not a regular file"; ::close(fd); fd = -1; return false;
        }
        size = (size_t) st_before.st_size;
        if (size && map_data) {
            void * m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m == MAP_FAILED) { error = path + ": mmap: " + strerror(errno); ::close(fd); fd = -1; return false; }
            ptr = (uint8_t *) m;
        }
        return true;
    }
    bool still_same(std::string & error) const {
        struct stat st{};
        if (fstat(fd, &st) != 0) { error = "fstat failed"; return false; }
        if (st.st_dev != st_before.st_dev || st.st_ino != st_before.st_ino || st.st_size != st_before.st_size ||
            st.st_mtim.tv_sec != st_before.st_mtim.tv_sec || st.st_mtim.tv_nsec != st_before.st_mtim.tv_nsec ||
            st.st_ctim.tv_sec != st_before.st_ctim.tv_sec || st.st_ctim.tv_nsec != st_before.st_ctim.tv_nsec) {
            error = "file changed while it was being read"; return false;
        }
        return true;
    }
    void close() {
        if (ptr) { munmap(ptr, size); ptr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    ~mapped_file() { close(); }
};

// Hash [ptr, ptr+size) sequentially. Memory-bound with the SHA extension; a whole model at ~1.5 GB/s per core.
std::string sha256_range(const uint8_t * ptr, size_t size) {
    sha256_ctx h; h.update(ptr, size); return h.hex();
}

// Run fn(i) for i in [0, n) on up to n_threads threads.
template <typename F>
void parallel_for(size_t n, int n_threads, F fn) {
    n_threads = std::max(1, std::min<int>(n_threads, (int) std::max<size_t>(1, n)));
    std::atomic<size_t> next{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < n_threads; ++t) {
        ts.emplace_back([&]() { for (size_t i; (i = next.fetch_add(1)) < n;) { fn(i); } });
    }
    for (auto & t : ts) { t.join(); }
}

// Strict JSON: the schema forbids duplicate keys, which nlohmann silently collapses. Track keys per open object.
bool parse_strict(const std::string & text, json & out, std::string & error) {
    std::vector<std::set<std::string>> scopes;
    bool dup = false;
    std::string dup_key;
    auto cb = [&](int /*depth*/, json::parse_event_t ev, json & parsed) -> bool {
        if (ev == json::parse_event_t::object_start) { scopes.emplace_back(); }
        else if (ev == json::parse_event_t::object_end) { if (!scopes.empty()) { scopes.pop_back(); } }
        else if (ev == json::parse_event_t::key) {
            const std::string k = parsed.get<std::string>();
            if (!scopes.empty() && !scopes.back().insert(k).second) { dup = true; dup_key = k; }
        }
        return true;
    };
    try {
        out = json::parse(text, cb, true);
    } catch (const std::exception & e) {
        error = std::string("manifest.json: ") + e.what();
        return false;
    }
    if (dup) { error = "manifest.json: duplicate key '" + dup_key + "'"; return false; }
    return true;
}

bool keys_exactly(const json & o, std::initializer_list<const char *> keys, const char * what, std::string & error) {
    if (!o.is_object()) { error = std::string(what) + " must be an object"; return false; }
    std::set<std::string> want(keys.begin(), keys.end()), got;
    for (auto it = o.begin(); it != o.end(); ++it) { got.insert(it.key()); }
    if (got != want) {
        error = std::string(what) + " must contain exactly {";
        for (auto k : keys) { error += k; error += ","; }
        error += "}";
        return false;
    }
    return true;
}

bool get_nonneg(const json & v, int64_t & out, const char * what, std::string & error) {
    if (!v.is_number_integer() || v.is_boolean()) { error = std::string(what) + " must be an integer"; return false; }
    const int64_t x = v.get<int64_t>();
    if (x < 0) { error = std::string(what) + " must be nonnegative"; return false; }
    out = x; return true;
}
bool get_positive(const json & v, int64_t & out, const char * what, std::string & error) {
    if (!get_nonneg(v, out, what, error)) { return false; }
    if (out == 0) { error = std::string(what) + " must be positive"; return false; }
    return true;
}
bool get_string(const json & v, std::string & out, const char * what, std::string & error) {
    if (!v.is_string()) { error = std::string(what) + " must be a string"; return false; }
    out = v.get<std::string>();
    if (out.empty()) { error = std::string(what) + " must be nonempty"; return false; }
    return true;
}

// Plane geometry from a record's own numbers: what the schema calls canonical shapes. bits/high_bits come from the
// record's arrangement (checked against the library by the loader); the packed-unit factor is derived from the
// units span the way quactlize derives it -- metadata bytes per superblock, paired when that is not a multiple of 4.
struct geometry {
    int64_t e1 = 1;                 // experts, or 1 for dense
    int64_t low_bytes = 0, high_bytes = 0, units_bytes = 0;
    int64_t spu = 1, ub = 0;        // superblocks per packed unit, unit bytes
    std::vector<int64_t> low_shape, high_shape, units_shape;
};

bool derive_geometry(int64_t n, int64_t k, int64_t experts, bool grouped, int32_t bits, int32_t high_bits,
                     int64_t units_bytes_total, geometry & g, std::string & error) {
    g.e1 = grouped ? experts : 1;
    if (n <= 0 || k <= 0 || g.e1 <= 0 || bits <= 0 || k % 256) { error = "invalid geometry"; return false; }
    g.low_bytes  = g.e1 * n * k * bits / 8;
    g.high_bytes = high_bits ? g.e1 * n * k * high_bits / 8 : 0;
    g.units_bytes = units_bytes_total;
    if (units_bytes_total <= 0 || units_bytes_total % g.e1) { error = "units plane does not divide by experts"; return false; }
    const int64_t units_e = units_bytes_total / g.e1;
    if ((units_e * 256) % (n * k)) { error = "units plane is not a whole number of bytes per superblock"; return false; }
    const int64_t sb = units_e * 256 / (n * k);
    g.spu = (sb % 4) ? 2 : 1;
    g.ub  = g.spu * sb;
    if (k % (256 * g.spu)) { error = "K is not a multiple of the packed-unit extent"; return false; }
    g.low_shape  = { g.e1, n, k * bits / 8 };
    g.high_shape = high_bits ? std::vector<int64_t>{ g.e1, n, k * high_bits / 8 } : std::vector<int64_t>{ 0 };
    g.units_shape = grouped ? std::vector<int64_t>{ experts, k / (256 * g.spu), n, g.ub }
                            : std::vector<int64_t>{ k / (256 * g.spu), n, g.ub };
    return true;
}

int64_t shape_product(const std::vector<int64_t> & s) {
    int64_t p = 1; for (auto x : s) { p *= x; } return p;
}

} // namespace

std::string llama_kpack_sha256_hex(const void * data, size_t size) {
    return sha256_range((const uint8_t *) data, size);
}

void llama_kpack_sha256_force_portable(bool on) { g_force_portable.store(on ? 1 : 0); }

// ============================================================================================================
// reader
// ============================================================================================================

struct llama_kpack_sidecar_reader::impl {
    json        manifest;
    mapped_file weights;
    std::string source_sha256;
    uint64_t    source_size = 0;
    std::string storage_sha256;
    uint64_t    storage_size = 0;
    bool        storage_verified = false;
};

llama_kpack_sidecar_reader::llama_kpack_sidecar_reader() : pimpl(new impl) {}
llama_kpack_sidecar_reader::~llama_kpack_sidecar_reader() = default;

const llama_kpack_sidecar_record * llama_kpack_sidecar_reader::find(const std::string & name) const {
    auto it = by_name.find(name);
    return it == by_name.end() ? nullptr : &records[it->second];
}

bool llama_kpack_sidecar_reader::open(const std::string & dir, std::string & error) {
    root = dir;
    struct stat st{};
    if (lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        error = "K-pack sidecar root must be a real directory: " + dir; return false;
    }
    // Exactly two entries, nothing else -- an unlisted file is a bundle that did not finish or was tampered with.
    {
        std::set<std::string> entries;
        DIR * d = opendir(dir.c_str());
        if (!d) { error = "cannot list " + dir; return false; }
        while (dirent * e = readdir(d)) {
            const std::string nm = e->d_name;
            if (nm != "." && nm != "..") { entries.insert(nm); }
        }
        closedir(d);
        if (entries != std::set<std::string>{ "manifest.json", "weights.bin" }) {
            error = "K-pack sidecar root entries disagree with the schema (want exactly manifest.json and weights.bin)";
            return false;
        }
    }

    std::string text;
    {
        mapped_file mf;
        if (!mf.open(dir + "/manifest.json", true, error)) { return false; }
        text.assign((const char *) mf.ptr, mf.size);
        if (!mf.still_same(error)) { return false; }
    }
    json & m = pimpl->manifest;
    if (!parse_strict(text, m, error)) { return false; }

    // ---- top level ----
    if (m.is_object() && m.value("schema", "") == KPACK_SCHEMA) {
        if (m.value("schema_version", 0) == 1) { error = "K-pack bundle schema v1 is source-unbound; repack it"; return false; }
        if (m.value("schema_version", 0) == 2) { error = "K-pack bundle schema v2 uses the retired NPY carrier; repack it"; return false; }
    }
    if (!keys_exactly(m, { "schema", "schema_version", "arrangement_version", "model", "selection", "source",
                           "storage", "tensors", "skipped" }, "K-pack manifest", error)) { return false; }
    if (m["schema"] != KPACK_SCHEMA || m["schema_version"] != KPACK_VERSION) {
        error = "unsupported K-pack bundle schema/version"; return false;
    }
    if (m["arrangement_version"] != 2) { error = "production K-pack bundles require placed arrangement version 2"; return false; }
    std::string model;
    if (!get_string(m["model"], model, "model", error)) { return false; }

    const json & src = m["source"];
    if (!keys_exactly(src, { "format", "size_bytes", "sha256" }, "source", error)) { return false; }
    if (src["format"] != "gguf") { error = "source.format must be gguf"; return false; }
    int64_t tmp;
    if (!get_positive(src["size_bytes"], tmp, "source.size_bytes", error)) { return false; }
    pimpl->source_size = (uint64_t) tmp;
    if (!src["sha256"].is_string() || !is_hex64(src["sha256"].get<std::string>())) { error = "source.sha256 must be lowercase hex"; return false; }
    pimpl->source_sha256 = src["sha256"].get<std::string>();

    const json & sto = m["storage"];
    if (!keys_exactly(sto, { "file", "size_bytes", "alignment_bytes", "sha256" }, "storage", error)) { return false; }
    if (sto["file"] != "weights.bin") { error = "storage.file must be weights.bin"; return false; }
    if (!get_positive(sto["size_bytes"], tmp, "storage.size_bytes", error)) { return false; }
    pimpl->storage_size = (uint64_t) tmp;
    if (sto["alignment_bytes"] != KPACK_ALIGN) { error = "storage alignment must be 128"; return false; }
    if (!sto["sha256"].is_string() || !is_hex64(sto["sha256"].get<std::string>())) { error = "storage.sha256 must be lowercase hex"; return false; }
    pimpl->storage_sha256 = sto["sha256"].get<std::string>();

    const json & sel = m["selection"];
    if (!keys_exactly(sel, { "layout_policy", "packable_total", "packed", "skipped" }, "selection", error)) { return false; }
    if (sel["layout_policy"] != "production-kpack-only") { error = "layout_policy must be production-kpack-only"; return false; }
    int64_t packable_total, packed, nskipped;
    if (!get_nonneg(sel["packable_total"], packable_total, "selection.packable_total", error) ||
        !get_nonneg(sel["packed"], packed, "selection.packed", error) ||
        !get_nonneg(sel["skipped"], nskipped, "selection.skipped", error)) { return false; }
    if (!m["tensors"].is_array() || m["tensors"].empty()) { error = "a production K-pack bundle must contain at least one tensor"; return false; }
    if (!m["skipped"].is_array()) { error = "skipped must be a list"; return false; }
    if (packed != (int64_t) m["tensors"].size()) { error = "selection.packed disagrees with tensors length"; return false; }
    if (nskipped != (int64_t) m["skipped"].size()) { error = "selection.skipped disagrees with skipped length"; return false; }
    if (packable_total != packed) { error = "selection.packable_total must equal packed for a complete bundle"; return false; }

    std::set<std::string> skipped_names;
    for (const auto & r : m["skipped"]) {
        if (!keys_exactly(r, { "name", "type_name", "reason" }, "skipped record", error)) { return false; }
        std::string nm, tn, why;
        if (!get_string(r["name"], nm, "skipped.name", error) || !get_string(r["type_name"], tn, "skipped.type_name", error) ||
            !get_string(r["reason"], why, "skipped.reason", error)) { return false; }
        if (!skipped_names.insert(nm).second) { error = "duplicate tensor name in skipped: " + nm; return false; }
    }

    // ---- records ----
    records.clear(); by_name.clear();
    uint64_t expected_region = 0;
    int64_t  prev_index = -1;
    uint64_t prev_end = 0;
    for (size_t i = 0; i < m["tensors"].size(); ++i) {
        const json & r = m["tensors"][i];
        llama_kpack_sidecar_record rec;
        const std::string where = "tensor record " + std::to_string(i);
        if (!keys_exactly(r, { "name", "ggml_type", "type_name", "route_class", "layout_name", "plane_packs", "rank",
                               "n", "k", "experts", "arrangement_version", "arrangement", "source_tensor", "region",
                               "spans" }, where.c_str(), error)) { return false; }
        if (!get_string(r["name"], rec.name, "name", error)) { return false; }
        const std::string pre = "artifact " + rec.name + ": ";
        int64_t t;
        if (!get_nonneg(r["ggml_type"], t, "ggml_type", error)) { return false; }
        rec.ggml_type = (int32_t) t;
        const char * tname = kquant_type_name(rec.ggml_type);
        if (!tname) { error = pre + "not a production K-pack qtype"; return false; }
        if (r["type_name"] != tname) { error = pre + "type_name disagrees with ggml_type"; return false; }
        if (!get_string(r["route_class"], rec.route_class, "route_class", error)) { return false; }
        const bool grouped = rec.route_class == "grouped";
        if (!grouped && rec.route_class != "dense") { error = pre + "route_class must be dense or grouped"; return false; }
        int64_t rank;
        if (!get_nonneg(r["rank"], rank, "rank", error)) { return false; }
        if ((grouped && rank != 3) || (!grouped && rank != 2)) { error = pre + "route_class/rank disagree"; return false; }
        if (!get_positive(r["n"], rec.n, "n", error) || !get_positive(r["k"], rec.k, "k", error)) { return false; }
        if (grouped) {
            if (!get_positive(r["experts"], rec.experts, "experts", error)) { return false; }
        } else {
            if (!r["experts"].is_null()) { error = pre + "dense tensor must record experts=null"; return false; }
            rec.experts = 0;
        }
        if (rec.n % 256 || rec.k % 256 || ((rec.ggml_type == GGML_TYPE_Q3_K || rec.ggml_type == GGML_TYPE_Q6_K) && rec.k % 512)) {
            error = pre + "geometry outside the resident K-pack domain"; return false;
        }
        if (r["layout_name"] != layout_name_for(rec.ggml_type)) { error = pre + "layout_name is not canonical"; return false; }
        if (r["arrangement_version"] != 2) { error = pre + "requires arrangement_version=2"; return false; }

        const json & a = r["arrangement"];
        if (!keys_exactly(a, { "layout", "bits", "high_bits", "artifact_tile_k", "transport_tile_k", "group_size",
                               "reserved", "mapping_id" }, "arrangement", error)) { return false; }
        int64_t f[8];
        const char * fn[8] = { "layout", "bits", "high_bits", "artifact_tile_k", "transport_tile_k", "group_size", "reserved", "mapping_id" };
        for (int j = 0; j < 8; ++j) { if (!get_nonneg(a[fn[j]], f[j], fn[j], error)) { error = pre + error; return false; } }
        auto & pl = rec.planes;
        pl.layout = (int32_t) f[0]; pl.bits = (int32_t) f[1]; pl.high_bits = (int32_t) f[2]; pl.artifact_tile_k = (int32_t) f[3];
        pl.transport_tile_k = (int32_t) f[4]; pl.group_size = (int32_t) f[5]; pl.reserved = (int32_t) f[6]; pl.mapping_id = (uint64_t) f[7];
        if (pl.layout != layout_id_for(rec.ggml_type) || pl.artifact_tile_k != 0 || pl.reserved != 0 || pl.bits <= 0) {
            error = pre + "arrangement is not a canonical K-pack descriptor"; return false;
        }
        const json & pp = r["plane_packs"];
        if (!keys_exactly(pp, { "low", "high" }, "plane_packs", error)) { return false; }
        if (pp["low"] != 16 / pl.bits || pp["high"] != (pl.high_bits ? 16 / pl.high_bits : 0)) {
            error = pre + "plane_packs disagree with the arrangement"; return false;
        }

        const int64_t raw_bytes = (int64_t) ggml_row_size((ggml_type) rec.ggml_type, rec.k) * rec.n * (grouped ? rec.experts : 1);

        const json & s = r["source_tensor"];
        if (!keys_exactly(s, { "index", "data_offset", "size_bytes", "sha256", "binding_sha256" }, "source_tensor", error)) { return false; }
        int64_t sidx, soff, ssize;
        if (!get_nonneg(s["index"], sidx, "source_tensor.index", error) ||
            !get_nonneg(s["data_offset"], soff, "source_tensor.data_offset", error) ||
            !get_positive(s["size_bytes"], ssize, "source_tensor.size_bytes", error)) { return false; }
        if (ssize != raw_bytes) { error = pre + "source tensor size is not the canonical GGUF size"; return false; }
        std::string ssha, sbind;
        if (!get_string(s["sha256"], ssha, "source_tensor.sha256", error) || !is_hex64(ssha) ||
            !get_string(s["binding_sha256"], sbind, "source_tensor.binding_sha256", error) || !is_hex64(sbind)) {
            error = pre + "source_tensor digests must be lowercase hex"; return false;
        }
        if (sbind != binding_digest(rec.name, rec.ggml_type, (int32_t) rank, rec.n, rec.k, grouped ? rec.experts : 0,
                                    sidx, (uint64_t) soff, (uint64_t) ssize, ssha)) {
            error = pre + "source tensor binding disagrees with its tensor identity"; return false;
        }

        const json & rg = r["region"];
        if (!keys_exactly(rg, { "offset_bytes", "size_bytes" }, "region", error)) { return false; }
        int64_t roff, rsize;
        if (!get_nonneg(rg["offset_bytes"], roff, "region.offset_bytes", error) || !get_positive(rg["size_bytes"], rsize, "region.size_bytes", error)) { return false; }
        if (roff % KPACK_ALIGN || rsize % KPACK_ALIGN) { error = pre + "region must be 128-byte aligned"; return false; }
        rec.region_offset = (uint64_t) roff; rec.region_size = (uint64_t) rsize;

        const json & sp = r["spans"];
        if (!keys_exactly(sp, { "low", "high", "units" }, "spans", error)) { return false; }
        llama_kpack_sidecar_record::span * spans[3] = { &rec.low, &rec.high, &rec.units };
        const char * span_names[3] = { "low", "high", "units" };
        for (int j = 0; j < 3; ++j) {
            const json & v = sp[span_names[j]];
            if (!keys_exactly(v, { "offset_bytes", "size_bytes", "shape", "sha256" }, "span", error)) { return false; }
            int64_t o, z;
            if (!get_nonneg(v["offset_bytes"], o, "span.offset_bytes", error) || !get_nonneg(v["size_bytes"], z, "span.size_bytes", error)) { return false; }
            if (!v["shape"].is_array()) { error = pre + "span shape must be a list"; return false; }
            for (const auto & x : v["shape"]) {
                int64_t d; if (!get_nonneg(x, d, "shape", error)) { return false; }
                spans[j]->shape.push_back(d);
            }
            if (!v["sha256"].is_string() || !is_hex64(v["sha256"].get<std::string>())) { error = pre + "span sha256 must be lowercase hex"; return false; }
            spans[j]->offset = (uint64_t) o; spans[j]->size = (uint64_t) z; spans[j]->sha256 = v["sha256"].get<std::string>();
        }
        geometry g;
        if (!derive_geometry(rec.n, rec.k, rec.experts, grouped, pl.bits, pl.high_bits, (int64_t) rec.units.size, g, error)) {
            error = pre + error; return false;
        }
        const std::vector<int64_t> * want_shape[3] = { &g.low_shape, &g.high_shape, &g.units_shape };
        uint64_t cursor = 0;
        for (int j = 0; j < 3; ++j) {
            if (spans[j]->offset != align_up(cursor)) { error = pre + std::string(span_names[j]) + " span offset is not canonical"; return false; }
            if (spans[j]->shape != *want_shape[j]) { error = pre + std::string(span_names[j]) + " shape is not canonical"; return false; }
            if ((int64_t) spans[j]->size != shape_product(spans[j]->shape)) { error = pre + std::string(span_names[j]) + " size is not canonical"; return false; }
            cursor = spans[j]->offset + spans[j]->size;
        }
        if ((int64_t) rec.low.size != g.low_bytes || (int64_t) rec.high.size != g.high_bytes) { error = pre + "plane sizes disagree with the arrangement"; return false; }
        if (rec.region_size != align_up(cursor)) { error = pre + "region size does not cover its spans"; return false; }
        if ((int64_t) rec.region_size != raw_bytes) { error = pre + "region is not byte-neutral with its GGUF tensor"; return false; }

        // ordering across records
        if (by_name.count(rec.name)) { error = "duplicate tensor name in K-pack manifest: " + rec.name; return false; }
        if (rec.region_offset != expected_region) { error = pre + "region is not in canonical manifest order"; return false; }
        if (sidx <= prev_index) { error = "K-pack source tensor indices must be strictly increasing"; return false; }
        if ((uint64_t) soff < prev_end) { error = "K-pack source tensor byte ranges must be ordered and disjoint"; return false; }
        expected_region += rec.region_size;
        prev_index = sidx; prev_end = (uint64_t) soff + (uint64_t) ssize;

        pl.low_bytes = rec.low.size; pl.high_bytes = rec.high.size; pl.units_bytes = rec.units.size;
        by_name[rec.name] = records.size();
        records.push_back(std::move(rec));
    }
    for (const auto & rec : records) {
        if (skipped_names.count(rec.name)) { error = "K-pack tensors and skipped inventory overlap: " + rec.name; return false; }
    }
    if (expected_region != pimpl->storage_size) { error = "K-pack tensor regions do not cover storage.size_bytes exactly"; return false; }

    if (!pimpl->weights.open(dir + "/weights.bin", true, error)) { return false; }
    if (pimpl->weights.size != pimpl->storage_size) {
        error = "K-pack storage size mismatch: expected " + std::to_string(pimpl->storage_size) + " observed " + std::to_string(pimpl->weights.size);
        return false;
    }
    return true;
}

bool llama_kpack_sidecar_reader::verify_source(const std::string & gguf_path,
        const std::vector<llama_kpack_source_tensor> & inventory, int n_threads, std::string & error) {
    std::map<std::string, const llama_kpack_source_tensor *> inv;
    for (const auto & t : inventory) { inv[t.name] = &t; }

    // Every record must be a tensor the loader actually has, with the identity the manifest claims. Index, absolute
    // offset, raw size, type and shape all have to agree -- a bundle from a re-quantised or re-ordered file is not
    // this file's bundle even when its bytes happen to hash the same.
    for (const auto & rec : records) {
        auto it = inv.find(rec.name);
        if (it == inv.end()) { error = "sidecar tensor " + rec.name + " is not in the model"; return false; }
        const auto & t = *it->second;
        const json & s = pimpl->manifest["tensors"][by_name.at(rec.name)]["source_tensor"];
        const bool grouped = rec.route_class == "grouped";
        if (t.gguf_index != s["index"].get<int64_t>() || t.data_offset != s["data_offset"].get<uint64_t>() ||
            t.size_bytes != s["size_bytes"].get<uint64_t>() || t.ggml_type != rec.ggml_type ||
            t.rank != (grouped ? 3 : 2) || t.n != rec.n || t.k != rec.k || t.experts != (grouped ? rec.experts : 0)) {
            error = "sidecar tensor " + rec.name + ": identity (index/offset/size/type/shape) disagrees with the GGUF";
            return false;
        }
    }

    mapped_file mf;
    if (!mf.open(gguf_path, false, error)) { return false; }
    if (mf.size != pimpl->source_size) {
        error = "source GGUF size " + std::to_string(mf.size) + " != manifest " + std::to_string(pimpl->source_size); return false;
    }
    for (const auto & rec : records) {
        const json & s = pimpl->manifest["tensors"][by_name.at(rec.name)]["source_tensor"];
        const uint64_t off = s["data_offset"].get<uint64_t>(), sz = s["size_bytes"].get<uint64_t>();
        if (off + sz > mf.size) { error = "sidecar tensor " + rec.name + " range is outside the GGUF"; return false; }
    }

    // The whole-file digest is one sequential pass; the per-tensor digests run alongside it on the other cores.
    std::string whole;
    std::thread whole_thread([&]() { whole = sha256_range(mf.ptr, mf.size); });
    std::vector<std::string> bad(records.size());
    parallel_for(records.size(), std::max(1, n_threads - 1), [&](size_t i) {
        const auto & rec = records[i];
        const json & s = pimpl->manifest["tensors"][by_name.at(rec.name)]["source_tensor"];
        const std::string got = sha256_range(mf.ptr + s["data_offset"].get<uint64_t>(), s["size_bytes"].get<size_t>());
        if (got != s["sha256"].get<std::string>()) { bad[i] = rec.name; }
    });
    whole_thread.join();
    for (const auto & b : bad) { if (!b.empty()) { error = "K-pack source tensor mismatch for " + b; return false; } }
    if (whole != pimpl->source_sha256) { error = "K-pack bundle source mismatch: the GGUF is not the one this bundle was built from"; return false; }
    if (!mf.still_same(error)) { return false; }
    return true;
}

bool llama_kpack_sidecar_reader::verify_storage(int n_threads, std::string & error) {
    auto & w = pimpl->weights;
    std::string whole;
    std::thread whole_thread([&]() { whole = sha256_range(w.ptr, w.size); });

    // Every span's digest, and every byte between and after spans inside a region must be zero.
    std::vector<std::string> bad(records.size());
    parallel_for(records.size(), std::max(1, n_threads - 1), [&](size_t i) {
        const auto & rec = records[i];
        const uint8_t * base = w.ptr + rec.region_offset;
        const llama_kpack_sidecar_record::span * spans[3] = { &rec.low, &rec.high, &rec.units };
        uint64_t cursor = 0;
        for (int j = 0; j < 3; ++j) {
            for (uint64_t q = cursor; q < spans[j]->offset; ++q) { if (base[q]) { bad[i] = rec.name + ": padding must be zero"; return; } }
            if (spans[j]->size && sha256_range(base + spans[j]->offset, spans[j]->size) != spans[j]->sha256) {
                bad[i] = rec.name + ": span checksum mismatch"; return;
            }
            cursor = spans[j]->offset + spans[j]->size;
        }
        for (uint64_t q = cursor; q < rec.region_size; ++q) { if (base[q]) { bad[i] = rec.name + ": trailing padding must be zero"; return; } }
    });
    whole_thread.join();
    for (const auto & b : bad) { if (!b.empty()) { error = "K-pack storage: " + b; return false; } }
    if (whole != pimpl->storage_sha256) { error = "K-pack storage checksum mismatch"; return false; }
    if (!w.still_same(error)) { return false; }

    for (auto & rec : records) {
        const uint8_t * base = w.ptr + rec.region_offset;
        rec.planes.low   = base + rec.low.offset;
        rec.planes.high  = rec.high.size ? base + rec.high.offset : nullptr;
        rec.planes.units = base + rec.units.offset;
    }
    pimpl->storage_verified = true;
    return true;
}

// ============================================================================================================
// writer
// ============================================================================================================

struct llama_kpack_sidecar_writer::impl {
    std::string final_dir, staging;
    int         fd = -1;
    uint64_t    pos = 0;
    sha256_ctx  whole;                       // over everything written to weights.bin, padding included
    ojson       tensors = ojson::array();
    ojson       skipped = ojson::array();
    int64_t     prev_index = -1;
    uint64_t    prev_end = 0;
    std::set<std::string> names;
    mapped_file source;
    std::string source_path;

    bool write_all(const void * data, size_t size, std::string & error) {
        const uint8_t * p = (const uint8_t *) data;
        while (size) {
            const ssize_t n = ::write(fd, p, size);
            if (n <= 0) { if (n < 0 && errno == EINTR) { continue; } error = std::string("weights.bin: write: ") + (n == 0 ? "no progress" : strerror(errno)); return false; }
            whole.update(p, (size_t) n);
            p += n; size -= (size_t) n; pos += (uint64_t) n;
        }
        return true;
    }
    bool pad_to(uint64_t target, std::string & error) {
        static const uint8_t zeros[4096] = { 0 };
        while (pos < target) {
            if (!write_all(zeros, (size_t) std::min<uint64_t>(sizeof(zeros), target - pos), error)) { return false; }
        }
        return true;
    }
    void remove_staging() {
        if (staging.empty()) { return; }
        if (fd >= 0) { ::close(fd); fd = -1; }
        ::unlink((staging + "/weights.bin").c_str());
        ::unlink((staging + "/manifest.json").c_str());
        ::rmdir(staging.c_str());
        staging.clear();
    }
};

llama_kpack_sidecar_writer::llama_kpack_sidecar_writer() : pimpl(new impl) {}
llama_kpack_sidecar_writer::~llama_kpack_sidecar_writer() { abort(); }

size_t llama_kpack_sidecar_writer::packed() const { return pimpl->tensors.size(); }

void llama_kpack_sidecar_writer::abort() {
    pimpl->remove_staging();
    pimpl->source.close();
}

bool llama_kpack_sidecar_writer::bind_source(const std::string & path, std::string & error, int loader_fd) {
    if (pimpl->source.fd >= 0) { error = "source is already bound"; return false; }
    if (!pimpl->source.open(path, false, error, false)) { return false; }
    if (loader_fd >= 0) {
        struct stat st{};
        const auto & bound = pimpl->source.st_before;
        if (fstat(loader_fd, &st) != 0 || st.st_dev != bound.st_dev || st.st_ino != bound.st_ino ||
            st.st_size != bound.st_size || st.st_mtim.tv_sec != bound.st_mtim.tv_sec ||
            st.st_mtim.tv_nsec != bound.st_mtim.tv_nsec || st.st_ctim.tv_sec != bound.st_ctim.tv_sec ||
            st.st_ctim.tv_nsec != bound.st_ctim.tv_nsec) {
            error = "source path is not the loader's file"; pimpl->source.close(); return false;
        }
    }
    pimpl->source_path = path;
    return true;
}

bool llama_kpack_sidecar_writer::begin(const std::string & dir, std::string & error) {
    if (pimpl->fd >= 0 || !pimpl->staging.empty()) { error = "writer is already open"; return false; }
    struct stat st{};
    if (lstat(dir.c_str(), &st) == 0) { error = "refusing to overwrite existing output " + dir; return false; }
    if (errno != ENOENT) { error = dir + ": lstat: " + strerror(errno); return false; }
    const std::string staging = dir + ".partial." + std::to_string((long) getpid());
    if (mkdir(staging.c_str(), 0755) != 0) { error = staging + ": mkdir: " + strerror(errno); return false; }
    // Cleanup may only own a directory this writer created successfully.
    pimpl->final_dir = dir;
    pimpl->staging   = staging;
    pimpl->fd = ::open((pimpl->staging + "/weights.bin").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (pimpl->fd < 0) { error = std::string("weights.bin: ") + strerror(errno); pimpl->remove_staging(); return false; }
    return true;
}

void llama_kpack_sidecar_writer::skip(const std::string & name, const std::string & type_name, const std::string & reason) {
    pimpl->skipped.push_back(ojson{ { "name", name }, { "type_name", type_name }, { "reason", reason } });
}

bool llama_kpack_sidecar_writer::add(const llama_kpack_source_tensor & src, const void * source_data,
                                     const llama_kpack_planes & planes, std::string & error) {
    if (!source_data) { error = "source data is null"; return false; }
    const read_chunk read = [&](size_t offset, size_t, std::string & why) -> const uint8_t * {
        const uint8_t * ptrs[] = { planes.low, planes.high, planes.units };
        const size_t sizes[] = { planes.low_bytes, planes.high_bytes, planes.units_bytes };
        for (int j = 0; j < 3; ++j) {
            if (offset < sizes[j]) {
                if (!ptrs[j]) { why = "plane is null"; return nullptr; }
                return ptrs[j] + offset;
            }
            offset -= sizes[j];
        }
        why = "plane offset is out of range";
        return nullptr;
    };
    return add_record(src, sha256_range((const uint8_t *) source_data, src.size_bytes), planes,
                      read, 8 * 1024 * 1024, {}, error);
}

bool llama_kpack_sidecar_writer::add_stream(const llama_kpack_source_tensor & src, const llama_kpack_planes & planes,
        const read_chunk & read, size_t chunk_bytes, const cancelled & cancel, std::string & error) {
    const auto & source = pimpl->source;
    if (source.fd < 0 || src.data_offset > source.size || src.size_bytes > source.size - src.data_offset) {
        error = "source range is outside the bound file"; return false;
    }
    return add_record(src, "", planes, read, chunk_bytes, cancel, error);
}

bool llama_kpack_sidecar_writer::add_record(const llama_kpack_source_tensor & src, const std::string & src_sha,
        const llama_kpack_planes & planes, const read_chunk & read, size_t chunk_bytes,
        const cancelled & cancel, std::string & error) {
    auto & I = *pimpl;
    const std::string pre = src.name + ": ";
    if (I.fd < 0) { error = "writer is not open"; return false; }
    if (!read || chunk_bytes == 0) { error = pre + "invalid chunk reader"; return false; }
    const char * tname = kquant_type_name(src.ggml_type);
    if (!tname) { error = pre + "not a K-pack format"; return false; }
    const bool grouped = src.rank == 3;
    if (!(src.rank == 2 || src.rank == 3) || (grouped && src.experts <= 0) || (!grouped && src.experts != 0)) {
        error = pre + "rank/experts disagree"; return false;
    }
    if (src.gguf_index <= I.prev_index) { error = pre + "records must be added in increasing GGUF index order"; return false; }
    if (src.data_offset < I.prev_end) { error = pre + "source byte ranges must be ascending and disjoint"; return false; }
    if (!I.names.insert(src.name).second) { error = pre + "duplicate tensor"; return false; }

    geometry g;
    if (!derive_geometry(src.n, src.k, src.experts, grouped, planes.bits, planes.high_bits, (int64_t) planes.units_bytes, g, error)) {
        error = pre + error; return false;
    }
    if ((int64_t) planes.low_bytes != g.low_bytes || (int64_t) planes.high_bytes != g.high_bytes) {
        error = pre + "plane sizes disagree with the arrangement"; return false;
    }
    const uint64_t raw_bytes = ggml_row_size((ggml_type) src.ggml_type, src.k) * (uint64_t) src.n * (uint64_t) g.e1;
    if (src.size_bytes != raw_bytes) { error = pre + "source size is not the canonical GGUF size"; return false; }

    // region
    const uint64_t region_offset = align_up(I.pos);
    if (!I.pad_to(region_offset, error)) { return false; }
    const uint64_t  sizes[3] = { planes.low_bytes, planes.high_bytes, planes.units_bytes };
    const std::vector<int64_t> * shapes[3] = { &g.low_shape, &g.high_shape, &g.units_shape };
    const char * span_names[3] = { "low", "high", "units" };
    ojson spans = ojson::object();
    uint64_t cursor = 0;
    size_t resident_offset = 0;
    for (int j = 0; j < 3; ++j) {
        const uint64_t off = align_up(cursor);
        if (!I.pad_to(region_offset + off, error)) { return false; }
        sha256_ctx span_hash;
        for (size_t copied = 0; copied < sizes[j];) {
            if (cancel && cancel()) { error = "cache write cancelled"; return false; }
            const size_t count = std::min<size_t>(chunk_bytes, sizes[j] - copied);
            const uint8_t * data = read(resident_offset + copied, count, error);
            if (!data) { return false; }
            span_hash.update(data, count);
            if (!I.write_all(data, count, error)) { return false; }
            copied += count;
        }
        ojson shape = ojson::array();
        for (auto d : *shapes[j]) { shape.push_back(d); }
        spans[span_names[j]] = ojson{ { "offset_bytes", off }, { "size_bytes", sizes[j] }, { "shape", shape }, { "sha256", span_hash.hex() } };
        resident_offset += sizes[j];
        cursor = off + sizes[j];
    }
    const uint64_t region_size = align_up(cursor);
    if (!I.pad_to(region_offset + region_size, error)) { return false; }
    if (region_size != raw_bytes) { error = pre + "resident region is not byte-neutral with the GGUF tensor"; return false; }

    const std::string bind = binding_digest(src.name, src.ggml_type, src.rank, src.n, src.k, grouped ? src.experts : 0,
                                            src.gguf_index, src.data_offset, src.size_bytes, src_sha);
    ojson arr = ojson{
        { "layout", planes.layout }, { "bits", planes.bits }, { "high_bits", planes.high_bits },
        { "artifact_tile_k", planes.artifact_tile_k }, { "transport_tile_k", planes.transport_tile_k },
        { "group_size", planes.group_size }, { "reserved", planes.reserved }, { "mapping_id", planes.mapping_id },
    };
    ojson rec = ojson{
        { "name", src.name }, { "ggml_type", src.ggml_type }, { "type_name", tname },
        { "route_class", grouped ? "grouped" : "dense" }, { "layout_name", layout_name_for(src.ggml_type) },
        { "plane_packs", ojson{ { "low", 16 / planes.bits }, { "high", planes.high_bits ? 16 / planes.high_bits : 0 } } },
        { "rank", src.rank }, { "n", src.n }, { "k", src.k },
        { "experts", grouped ? ojson(src.experts) : ojson(nullptr) },
        { "arrangement_version", 2 }, { "arrangement", arr },
        { "source_tensor", ojson{ { "index", src.gguf_index }, { "data_offset", src.data_offset },
                                  { "size_bytes", src.size_bytes }, { "sha256", src_sha }, { "binding_sha256", bind } } },
        { "region", ojson{ { "offset_bytes", region_offset }, { "size_bytes", region_size } } },
        { "spans", spans },
    };
    I.tensors.push_back(std::move(rec));
    I.prev_index = src.gguf_index;
    I.prev_end   = src.data_offset + src.size_bytes;
    return true;
}

static bool fsync_path(const std::string & path, std::string & error) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { error = path + ": " + strerror(errno); return false; }
    const bool ok = fsync(fd) == 0;
    if (!ok) { error = path + ": fsync: " + strerror(errno); }
    ::close(fd);
    return ok;
}

bool llama_kpack_sidecar_writer::finish(const std::string & model_label, const std::string & gguf_path,
                                      std::string & error, const cancelled & cancel) {
    auto & I = *pimpl;
    if (I.fd < 0) { error = "writer is not open"; return false; }
    if (I.tensors.empty()) { error = "refusing to create an empty artifact bundle"; abort(); return false; }
    if (fsync(I.fd) != 0) { error = std::string("weights.bin: fsync: ") + strerror(errno); abort(); return false; }
    ::close(I.fd); I.fd = -1;

    // The source authority: the whole file, hashed now so that a file rewritten during the load is caught.
    std::string src_sha; uint64_t src_size = 0;
    {
        mapped_file temporary;
        if (I.source.fd < 0 && !temporary.open(gguf_path, false, error, false)) { abort(); return false; }
        const mapped_file & mf = I.source.fd >= 0 ? I.source : temporary;
        struct stat current{};
        if (stat(gguf_path.c_str(), &current) != 0 || current.st_dev != mf.st_before.st_dev ||
            current.st_ino != mf.st_before.st_ino || (I.source.fd >= 0 && gguf_path != I.source_path)) {
            error = "source path changed since load"; abort(); return false;
        }
        src_size = mf.size;
        if (!mf.still_same(error)) { abort(); return false; }
        // Legacy v3 source authority, independent of D2H. Read once and hash
        // whole-file and tensor ranges together; never retain raw tensor data.
        std::vector<uint8_t> chunk(1024 * 1024);
        std::vector<sha256_ctx> tensor_hashes(I.tensors.size());
        sha256_ctx file_hash;
        size_t first_tensor = 0;
        for (uint64_t off = 0; off < src_size;) {
            if (cancel && cancel()) { error = "cache write cancelled"; abort(); return false; }
            const size_t count = std::min<uint64_t>(chunk.size(), src_size - off);
            const ssize_t got = pread(mf.fd, chunk.data(), count, off);
            if (got < 0 && errno == EINTR) { continue; }
            if (got <= 0) { error = "source read failed or truncated"; abort(); return false; }
            const uint64_t end = off + (size_t) got;
            file_hash.update(chunk.data(), (size_t) got);
            for (size_t j = first_tensor; j < I.tensors.size(); ++j) {
                const auto & s = I.tensors[j]["source_tensor"];
                const uint64_t begin = s["data_offset"].get<uint64_t>();
                const uint64_t last = begin + s["size_bytes"].get<uint64_t>();
                if (last <= off) { first_tensor = j + 1; continue; }
                if (begin >= end) { break; }
                const uint64_t from = std::max(off, begin), to = std::min(end, last);
                tensor_hashes[j].update(chunk.data() + from - off, to - from);
            }
            off = end;
        }
        if (!mf.still_same(error)) { abort(); return false; }
        src_sha = file_hash.hex();
        for (size_t j = 0; j < I.tensors.size(); ++j) {
            auto & r = I.tensors[j];
            auto & s = r["source_tensor"];
            const uint64_t off = s["data_offset"].get<uint64_t>(), bytes = s["size_bytes"].get<uint64_t>();
            if (off > src_size || bytes > src_size - off) { error = "source range is outside file"; abort(); return false; }
            const std::string digest = tensor_hashes[j].hex();
            if (!s["sha256"].get<std::string>().empty() && s["sha256"] != digest) {
                error = "source bytes differ from the supplied tensor"; abort(); return false;
            }
            s["sha256"] = digest;
            s["binding_sha256"] = binding_digest(r["name"], r["ggml_type"], r["rank"], r["n"], r["k"],
                r["experts"].is_null() ? 0 : r["experts"].get<int64_t>(), s["index"], off, bytes, digest);
        }
    }
    if (src_size == 0) { error = "source GGUF must not be empty"; abort(); return false; }

    ojson manifest = ojson{
        { "schema", KPACK_SCHEMA }, { "schema_version", KPACK_VERSION }, { "arrangement_version", 2 },
        { "model", model_label },
        { "source", ojson{ { "format", "gguf" }, { "size_bytes", src_size }, { "sha256", src_sha } } },
        { "storage", ojson{ { "file", "weights.bin" }, { "size_bytes", I.pos }, { "alignment_bytes", KPACK_ALIGN }, { "sha256", I.whole.hex() } } },
        { "selection", ojson{ { "layout_policy", "production-kpack-only" }, { "packable_total", I.tensors.size() },
                              { "packed", I.tensors.size() }, { "skipped", I.skipped.size() } } },
        { "tensors", I.tensors }, { "skipped", I.skipped },
    };
    const std::string text = manifest.dump(2) + "\n";
    const int mfd = ::open((I.staging + "/manifest.json").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (mfd < 0) { error = std::string("manifest.json: ") + strerror(errno); abort(); return false; }
    {
        const char * p = text.data(); size_t left = text.size();
        while (left) {
            const ssize_t n = ::write(mfd, p, left);
            if (n < 0) { if (errno == EINTR) { continue; } error = std::string("manifest.json: write: ") + strerror(errno); ::close(mfd); abort(); return false; }
            p += n; left -= (size_t) n;
        }
        if (fsync(mfd) != 0) { error = "manifest.json: fsync failed"; ::close(mfd); abort(); return false; }
        ::close(mfd);
    }
    if (!fsync_path(I.staging, error)) { abort(); return false; }
    if (cancel && cancel()) { error = "cache write cancelled"; abort(); return false; }

    // Publish without ever replacing: the kernel's RENAME_NOREPLACE, not a racy existence check.
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)
#endif
    if (syscall(SYS_renameat2, AT_FDCWD, I.staging.c_str(), AT_FDCWD, I.final_dir.c_str(), RENAME_NOREPLACE) != 0) {
        error = "refusing to overwrite existing output " + I.final_dir + " (" + strerror(errno) + ")";
        abort(); return false;
    }
    I.staging.clear();
    I.source.close();
    std::string parent = I.final_dir;
    const size_t slash = parent.find_last_of('/');
    parent = slash == std::string::npos ? "." : (slash == 0 ? "/" : parent.substr(0, slash));
    fsync_path(parent, error);
    error.clear();
    GGML_LOG_INFO("[kpack] wrote %zu artifact(s) to %s\n", I.tensors.size(), I.final_dir.c_str());
    return true;
}

struct llama_kpack_background_writer::impl {
    llama_kpack_sidecar_writer writer;
    std::string source_path, error;
    std::thread worker;
    std::atomic<bool> cancelled{false};
    bool prepared = false, started = false, success = false;
};

llama_kpack_background_writer::llama_kpack_background_writer() : pimpl(new impl) {}
llama_kpack_background_writer::~llama_kpack_background_writer() { cancel(); }

bool llama_kpack_background_writer::prepare(const std::string & dir, const std::string & source_path,
                                            std::string & error, int loader_fd) {
    auto & I = *pimpl;
    if (I.prepared || I.started) { error = "background writer is already prepared"; return false; }
    if (!I.writer.bind_source(source_path, error, loader_fd) || !I.writer.begin(dir, error)) {
        I.writer.abort(); return false;
    }
    I.source_path = source_path;
    I.prepared = true;
    return true;
}

bool llama_kpack_background_writer::start(std::vector<llama_kpack_write_job> jobs,
        const std::vector<llama_kpack_source_tensor> & inventory, size_t chunk_bytes, std::string & error) {
    auto & I = *pimpl;
    if (!I.prepared || I.started || I.cancelled || jobs.empty() || chunk_bytes == 0) {
        error = "background writer cannot start"; return false;
    }
    std::sort(jobs.begin(), jobs.end(), [](const llama_kpack_write_job & a, const llama_kpack_write_job & b) {
        return a.source.gguf_index < b.source.gguf_index;
    });
    std::set<std::string> names;
    for (const auto & job : jobs) {
        if (!job.read || !names.insert(job.source.name).second) { error = "invalid or duplicate snapshot"; return false; }
    }
    for (const auto & src : inventory) {
        if (!names.count(src.name)) {
            I.writer.skip(src.name, ggml_type_name((ggml_type) src.ggml_type), "not resident in a K-pack buffer");
        }
    }
    try {
        I.worker = std::thread([&I, jobs = std::move(jobs), chunk_bytes]() {
            const auto cancel = [&I]() { return I.cancelled.load(); };
            try {
                for (const auto & job : jobs) {
                    if (!I.writer.add_stream(job.source, job.planes, job.read, chunk_bytes, cancel, I.error)) {
                        I.writer.abort(); return;
                    }
                }
                I.success = I.writer.finish(I.source_path, I.source_path, I.error, cancel);
            } catch (const std::exception & e) {
                I.error = e.what();
            } catch (...) {
                I.error = "unknown background cache failure";
            }
            if (!I.success) { I.writer.abort(); }
        });
    } catch (const std::exception & e) {
        error = e.what(); I.writer.abort(); return false;
    }
    I.started = true;
    return true;
}

bool llama_kpack_background_writer::wait(std::string & error) {
    auto & I = *pimpl;
    if (I.worker.joinable()) { I.worker.join(); }
    error = I.error;
    return I.started && I.success;
}

void llama_kpack_background_writer::cancel() {
    pimpl->cancelled = true;
    if (pimpl->worker.joinable()) { pimpl->worker.join(); }
    pimpl->writer.abort();
}
