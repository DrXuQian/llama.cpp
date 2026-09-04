// The persistent K-pack sidecar, exercised without a device: a GGUF is written with the gguf API, planes come from
// the stub quactlize libraries, the writer publishes a bundle, the reader validates it, and every guard is shown to
// fire on a bundle that is wrong in exactly one way. With KPACK_PY / KPACK_PY_REPO set, quactlize's own
// load_kpack_bundle is run on the same bundle -- the interoperability check that matters, since the format is theirs.

#include "llama-kpack-sidecar.h"
#include "quactlize-lib.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...) do { \
    const bool ok_ = (cond); \
    printf("    %-70s %s\n", #cond, ok_ ? "ok" : "FAIL"); \
    if (!ok_) { g_failures++; printf("      "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static std::vector<uint8_t> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void write_file(const std::string & p, const std::vector<uint8_t> & d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write((const char *) d.data(), (std::streamsize) d.size());
}
static void copy_bundle(const std::string & src, const std::string & dst) {
    mkdir(dst.c_str(), 0755);
    write_file(dst + "/manifest.json", read_file(src + "/manifest.json"));
    write_file(dst + "/weights.bin",   read_file(src + "/weights.bin"));
}
static void rm_bundle(const std::string & d) {
    unlink((d + "/manifest.json").c_str()); unlink((d + "/weights.bin").c_str()); unlink((d + "/extra").c_str()); rmdir(d.c_str());
}

// ---- SHA-256 ----
static void test_sha256() {
    printf("  [sha256]\n");
    const std::string abc = "abc";
    std::vector<uint8_t> million(1000000, 'a');
    for (int portable = 0; portable < 2; ++portable) {
        llama_kpack_sha256_force_portable(portable);
        CHECK(llama_kpack_sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "portable=%d", portable);
        CHECK(llama_kpack_sha256_hex(abc.data(), 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "portable=%d", portable);
        CHECK(llama_kpack_sha256_hex(million.data(), million.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "portable=%d", portable);
    }
    // The two paths on random lengths: the only reason to trust the SHA-extension schedule.
    unsigned seed = 42; int mism = 0;
    for (int i = 0; i < 200; ++i) {
        seed = seed*1103515245u + 12345u;
        const size_t len = (seed >> 8) % 1500;
        std::vector<uint8_t> b(len);
        for (auto & x : b) { seed = seed*1103515245u + 12345u; x = (uint8_t) (seed >> 16); }
        llama_kpack_sha256_force_portable(true);  const std::string a = llama_kpack_sha256_hex(b.data(), len);
        llama_kpack_sha256_force_portable(false); const std::string c = llama_kpack_sha256_hex(b.data(), len);
        if (a != c) { mism++; }
    }
    CHECK(mism == 0, "portable vs accelerated mismatches: %d", mism);
    llama_kpack_sha256_force_portable(false);
}

struct made_tensor {
    std::string name; int qtype; int64_t n, k, experts; int rank;
    std::vector<uint8_t> data;            // GGUF bytes
    std::vector<uint8_t> low, high, units;
    llama_kpack_planes planes;
    llama_kpack_source_tensor src;
};

static bool make_planes(made_tensor & t) {
    quactlize_ppu_placed_arrangement_v2 arr;
    if (!ggml_quactlize_arrangement_for(t.qtype, &arr)) { return false; }
    const int64_t e1 = t.rank == 3 ? t.experts : 1;
    int64_t lo, hi, un;
    if (!ggml_quactlize_plane_sizes(t.qtype, t.n, t.k, e1, &arr, &lo, &hi, &un)) { return false; }
    t.low.resize((size_t) lo); t.high.resize((size_t) (hi ? hi : 1)); t.units.resize((size_t) un);
    if (ggml_quactlize_prepare(t.qtype, t.data.data(), t.low.data(), hi ? t.high.data() : nullptr, t.units.data(),
                               (int) t.n, (int) t.k, (int) e1, &arr) != 0) { return false; }
    auto & p = t.planes;
    p.low = t.low.data(); p.low_bytes = (size_t) lo;
    p.high = hi ? t.high.data() : nullptr; p.high_bytes = (size_t) hi;
    p.units = t.units.data(); p.units_bytes = (size_t) un;
    p.layout = arr.layout; p.bits = arr.bits; p.high_bits = arr.high_bits; p.artifact_tile_k = arr.artifact_tile_k;
    p.transport_tile_k = arr.transport_tile_k; p.group_size = arr.group_size; p.reserved = arr.reserved; p.mapping_id = arr.mapping_id;
    return true;
}

int main() {
    test_sha256();

    // ---- a GGUF with one dense and one grouped Q4_K tensor, plus one that is not packable ----
    printf("  [gguf]\n");
    std::string base = getenv("KPACK_TEST_DIR") ? getenv("KPACK_TEST_DIR") : "/tmp";
    char tmpl[4096]; snprintf(tmpl, sizeof(tmpl), "%s/kpack-test-XXXXXX", base.c_str());
    const std::string dir = mkdtemp(tmpl);
    const std::string gguf_path = dir + "/model.gguf";

    struct ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context * gctx = ggml_init(ip);
    made_tensor dense  { "blk.0.attn_q.weight",        GGML_TYPE_Q4_K, 256, 512, 0, 2, {}, {}, {}, {}, {}, {} };
    made_tensor grouped{ "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q4_K, 256, 512, 4, 3, {}, {}, {}, {}, {}, {} };
    ggml_tensor * t_norm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 256); ggml_set_name(t_norm, "blk.0.attn_norm.weight");
    ggml_tensor * t_dense = ggml_new_tensor_2d(gctx, GGML_TYPE_Q4_K, dense.k, dense.n); ggml_set_name(t_dense, dense.name.c_str());
    ggml_tensor * t_group = ggml_new_tensor_3d(gctx, GGML_TYPE_Q4_K, grouped.k, grouped.n, grouped.experts); ggml_set_name(t_group, grouped.name.c_str());
    unsigned seed = 7;
    for (ggml_tensor * t : { t_norm, t_dense, t_group }) {
        uint8_t * d = (uint8_t *) t->data;
        for (size_t i = 0; i < ggml_nbytes(t); ++i) { seed = seed*1103515245u + 12345u; d[i] = (uint8_t) (seed >> 16); }
    }
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_add_tensor(g, t_norm);     // index 0: not packable
    gguf_add_tensor(g, t_dense);    // index 1
    gguf_add_tensor(g, t_group);    // index 2
    gguf_write_to_file(g, gguf_path.c_str(), false);

    auto fill = [&](made_tensor & m, ggml_tensor * t, int index) {
        m.data.assign((uint8_t *) t->data, (uint8_t *) t->data + ggml_nbytes(t));
        m.src.name = m.name; m.src.gguf_index = index;
        // gguf_get_data_offset() is only meaningful for a context read FROM a file; for one being written, the data
        // section starts right after the metadata, which is what gguf_get_meta_size() reports.
        m.src.data_offset = gguf_get_meta_size(g) + gguf_get_tensor_offset(g, index);
        m.src.size_bytes = ggml_nbytes(t); m.src.ggml_type = m.qtype; m.src.rank = m.rank;
        m.src.n = m.n; m.src.k = m.k; m.src.experts = m.rank == 3 ? m.experts : 0;
    };
    fill(dense, t_dense, 1); fill(grouped, t_group, 2);
    CHECK(make_planes(dense),   "stub library must supply planes for the dense tensor");
    CHECK(make_planes(grouped), "stub library must supply planes for the grouped tensor");
    std::vector<llama_kpack_source_tensor> inventory = { dense.src, grouped.src };

    // ---- writer ----
    printf("  [writer]\n");
    const std::string bundle = gguf_path + ".kpack";
    std::string err;
    {
        llama_kpack_sidecar_writer w;
        CHECK(w.begin(bundle, err), "%s", err.c_str());
        w.skip("blk.0.attn_norm.weight", "F32", "not a k-quant this build packs");
        CHECK(w.add(dense.src, dense.data.data(), dense.planes, err), "%s", err.c_str());
        // out of order must be refused: a record with a lower index after a higher one
        llama_kpack_source_tensor bad = dense.src; bad.name = "x"; bad.gguf_index = 0;
        CHECK(!w.add(bad, dense.data.data(), dense.planes, err), "index order guard");
        CHECK(w.add(grouped.src, grouped.data.data(), grouped.planes, err), "%s", err.c_str());
        CHECK(w.finish(gguf_path, gguf_path, err), "%s", err.c_str());
        CHECK(w.packed() == 2, "packed=%zu", w.packed());
    }
    struct stat st{};
    CHECK(stat((bundle + "/manifest.json").c_str(), &st) == 0 && stat((bundle + "/weights.bin").c_str(), &st) == 0, "bundle files exist");
    {
        llama_kpack_sidecar_writer w2;
        CHECK(!w2.begin(bundle, err), "a second writer must refuse the existing bundle");
    }

    // ---- reader ----
    printf("  [reader]\n");
    {
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err), "%s", err.c_str());
        CHECK(r.verify_source(gguf_path, inventory, 4, err), "%s", err.c_str());
        CHECK(r.verify_storage(4, err), "%s", err.c_str());
        const auto * rd = r.find(dense.name); const auto * rg = r.find(grouped.name);
        CHECK(rd && rg && !r.find("nope"), "records");
        if (rd && rg) {
            CHECK(rd->planes.low_bytes == dense.planes.low_bytes && memcmp(rd->planes.low, dense.low.data(), dense.planes.low_bytes) == 0, "dense low plane round-trips");
            CHECK(rd->planes.units_bytes == dense.planes.units_bytes && memcmp(rd->planes.units, dense.units.data(), dense.planes.units_bytes) == 0, "dense units plane round-trips");
            CHECK(rg->planes.low_bytes == grouped.planes.low_bytes && memcmp(rg->planes.low, grouped.low.data(), grouped.planes.low_bytes) == 0, "grouped low plane round-trips");
            CHECK(rg->planes.high == nullptr && rg->planes.high_bytes == 0, "Q4_K has no high plane");
            CHECK(rd->planes.mapping_id == dense.planes.mapping_id && rd->planes.bits == 4, "arrangement carried");
            CHECK(rg->units.shape == std::vector<int64_t>({ 4, 2, 256, 16 }), "grouped units shape [E, K/256, N, 16]");
            CHECK(rd->units.shape == std::vector<int64_t>({ 2, 256, 16 }), "dense units shape [K/256, N, 16]");
            CHECK(rd->low.shape == std::vector<int64_t>({ 1, 256, 256 }), "dense low shape [1, N, K/2]");
        }
    }

    // ---- negatives: one wrong thing each ----
    printf("  [negatives]\n");
    {   // a flipped byte inside a span
        const std::string c = dir + "/c1"; copy_bundle(bundle, c);
        auto w = read_file(c + "/weights.bin"); w[1000] ^= 0x01; write_file(c + "/weights.bin", w);
        llama_kpack_sidecar_reader r;
        CHECK(r.open(c, err) && !r.verify_storage(4, err), "flipped weights byte must fail storage verification (%s)", err.c_str());
        rm_bundle(c);
    }
    {   // an extra file in the root
        const std::string c = dir + "/c2"; copy_bundle(bundle, c); write_file(c + "/extra", { 1 });
        llama_kpack_sidecar_reader r;
        CHECK(!r.open(c, err), "extra root entry must be refused");
        rm_bundle(c);
    }
    {   // a duplicate key in the manifest
        const std::string c = dir + "/c3"; copy_bundle(bundle, c);
        auto m = read_file(c + "/manifest.json"); std::string s(m.begin(), m.end());
        s.insert(s.find("\"schema\""), "\"schema\": \"x\",\n  "); write_file(c + "/manifest.json", std::vector<uint8_t>(s.begin(), s.end()));
        llama_kpack_sidecar_reader r;
        CHECK(!r.open(c, err), "duplicate manifest key must be refused");
        rm_bundle(c);
    }
    {   // the wrong GGUF: one byte of tensor data changed
        auto gb = read_file(gguf_path); gb[dense.src.data_offset + 77] ^= 0x80;
        const std::string other = dir + "/other.gguf"; write_file(other, gb);
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err) && !r.verify_source(other, inventory, 4, err), "a modified source must fail (%s)", err.c_str());
        unlink(other.c_str());
    }
    {   // an inventory that disagrees on offset
        auto inv2 = inventory; inv2[0].data_offset += 32;
        llama_kpack_sidecar_reader r;
        CHECK(r.open(bundle, err) && !r.verify_source(gguf_path, inv2, 4, err), "inventory/offset disagreement must fail");
    }

    // ---- quactlize's own validator on the same bundle ----
    printf("  [interop]\n");
    const char * py = getenv("KPACK_PY"); const char * repo = getenv("KPACK_PY_REPO");
    if (py && repo) {
        auto run = [&](const std::string & b, const std::string & src) {
            // No pipe: system() returns the LAST command's status, and a `| tail` would make every run look like 0.
            std::string cmd = "cd '" + std::string(repo) + "' && '" + py + "' -c \"from quactlize.pack_gguf import load_kpack_bundle; "
                "b=load_kpack_bundle('" + b + "', source='" + src + "'); print('PY-OK', len(b.artifacts))\" > /dev/null 2>&1";
            return system(cmd.c_str());
        };
        CHECK(run(bundle, gguf_path) == 0, "quactlize load_kpack_bundle must accept the bundle");
        const std::string c = dir + "/c4"; copy_bundle(bundle, c);
        auto w = read_file(c + "/weights.bin"); w[2000] ^= 0x01; write_file(c + "/weights.bin", w);
        CHECK(run(c, gguf_path) != 0, "quactlize must reject the flipped-byte copy (the oracle can say no)");
        rm_bundle(c);
    } else {
        printf("    (KPACK_PY / KPACK_PY_REPO not set: quactlize cross-check skipped)\n");
    }

    rm_bundle(bundle);
    unlink(gguf_path.c_str()); rmdir(dir.c_str());
    gguf_free(g); ggml_free(gctx);
    printf("test-kpack-sidecar: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
