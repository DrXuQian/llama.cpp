// The llama.cpp half of the quactlize K-pack loader, against stub libraries.
//
// WHY THIS EXISTS AND WHY IT RUNS OFF-BOX. The guards under test decide whether a tensor's GGUF bytes get replaced
// by a K-pack artifact -- after which there is no second copy and no fallback. They are therefore the guards most
// worth having a negative control for, and they are also the only part of the path that needs neither a PPU device
// nor a real kernel library: identity, the format-registry cross-check, capability-vs-presence, and the byte
// neutrality the whole design rests on.
//
// The loader arms itself once per process (pthread_once), so one case per process: with no --case the binary is the
// driver and re-execs itself, once per row of the table, with that row's environment.

#include "quactlize-lib.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

enum qz_kind {
    QZ_GATES,     // the admission guards: identity, registry, capability, byte neutrality
    QZ_CONVERT,   // the conversion itself: threading, the round trip, and the fallback when the split is invalid
};

struct qz_case {
    qz_kind      kind;
    const char * name;
    const char * env;              // what makes the stub lie, "" for the honest library
    int          qtype;
    bool         want_available;
    bool         want_arrangement;
    bool         want_conversion;
    // Whether the byte-neutrality identity is expected to HOLD. False is a case that exists to prove the identity
    // can be broken -- without one, "the sums always matched" says nothing about whether anything is checking.
    bool         want_byte_neutral;
    bool         hide_bundle;      // run with an empty loader path: nothing installed
    const char * why;
};

// Five positives, one per format; five negatives, each making one guard fire; two deployment conditions. Every
// negative must flip at least one column to false: a guard that cannot be made to fire is not a guard.
static const qz_case g_cases[] = {
    { QZ_GATES, "q2k",  "", GGML_TYPE_Q2_K, true,  true,  true,  true,  false, "fmt2 -> Q2_K, arrangement matches the registry" },
    { QZ_GATES, "q3k",  "", GGML_TYPE_Q3_K, true,  true,  true,  true,  false, "fmt3 -> Q3_K, two planes (2 + 1 bits)" },
    { QZ_GATES, "q4k",  "", GGML_TYPE_Q4_K, true,  true,  true,  true,  false, "fmt0 -> Q4_K, one plane, gs=32" },
    { QZ_GATES, "q5k",  "", GGML_TYPE_Q5_K, true,  true,  true,  true,  false, "fmt1 -> Q5_K, two planes (4 + 1 bits)" },
    { QZ_GATES, "q6k",  "", GGML_TYPE_Q6_K, true,  true,  true,  true,  false, "fmt4 -> Q6_K, two planes (4 + 2 bits)" },

    { QZ_GATES, "wrong-identity",    "QZ_STUB_FMT=3",           GGML_TYPE_Q4_K, false, false, false, true,  false,
      "the file named fmt0 reports format 3: a reshuffled bundle must not arm" },
    { QZ_GATES, "default-library",   "QZ_STUB_NO_IDENTITY=1",   GGML_TYPE_Q4_K, false, false, false, true,  false,
      "the default/ScaleFirst build reports -1 and is not a K-pack library" },
    { QZ_GATES, "registry-mismatch", "QZ_STUB_BAD_GS=1",        GGML_TYPE_Q4_K, true,  false, false, true,  false,
      "group_size off by one against ppu_format_config.inc" },
    { QZ_GATES, "xplane-descriptor", "QZ_STUB_BAD_ATK=1",       GGML_TYPE_Q4_K, true,  false, false, true,  false,
      "artifact_tile_k != 0: an Xplane descriptor arriving by the K-pack door" },
    { QZ_GATES, "no-conversion",     "QZ_STUB_NO_CONVERSION=1", GGML_TYPE_Q4_K, true,  true,  false, true,  false,
      "the conversion entries are exported but cannot answer for a 256-code superblock" },

    { QZ_GATES, "bad-units",         "QZ_STUB_BAD_UNITS=1",     GGML_TYPE_Q4_K, true,  true,  true,  false, false,
      "units_bytes one superblock too large: the descriptor still matches, only byte neutrality can catch it" },

    { QZ_GATES, "unsupported-type",  "",                        GGML_TYPE_Q8_0, false, false, false, true,  false,
      "Q8_0 is outside the K-pack format range: unavailable, not a lookup past the table" },
    { QZ_GATES, "no-bundle",         "",                        GGML_TYPE_Q4_K, false, false, false, true,  true,
      "nothing installed on the loader path: every format unavailable, no crash" },

    // The conversion path. want_available doubles as "expect rc == 0"; want_arrangement as "expect more than one
    // thread to have produced the accepted result". The stub's packing is slice-sensitive on purpose, so a
    // threaded run that lands bytes at the wrong expert offset cannot pass.
    { QZ_CONVERT, "convert-threaded", "",                          GGML_TYPE_Q4_K, true,  true,  true,  true, false,
      "eight experts split across threads, round-tripping" },
    { QZ_CONVERT, "convert-serial",   "GGML_QUACTLIZE_CONVERT_THREADS=1", GGML_TYPE_Q4_K, true, false, true, true, false,
      "the knob that makes the threaded-vs-serial load delta measurable" },
    { QZ_CONVERT, "convert-fallback", "QZ_STUB_INTERLEAVE=1",      GGML_TYPE_Q4_K, true,  false, true,  true, false,
      "artifact is NOT a per-expert concatenation: the split must fail, be detected, and fall back to serial" },
};

static const size_t g_ncases = sizeof(g_cases) / sizeof(g_cases[0]);

// Shapes the byte-neutrality identity has to hold on. K is a multiple of 256 because that is a k-quant superblock;
// the first row is the expert shape actually measured on the box.
static const struct { int64_t n, k, experts; } g_shapes[] = {
    {   512, 3072, 256 },
    {  4096, 4096,   1 },
    {     1,  256,   1 },
    { 14336,  512,   8 },
};

static const size_t g_nshapes = sizeof(g_shapes) / sizeof(g_shapes[0]);

static int check_byte_neutrality(int qtype, const quactlize_ppu_placed_arrangement_v2 & arr) {
    int failures = 0;
    for (size_t s = 0; s < g_nshapes; ++s) {
        const int64_t n = g_shapes[s].n, k = g_shapes[s].k, e = g_shapes[s].experts;

        int64_t low = 0, high = 0, units = 0;
        if (!ggml_quactlize_plane_sizes(qtype, n, k, e, &arr, &low, &high, &units)) {
            printf("      FAIL: no plane sizes for n=%lld k=%lld e=%lld\n",
                   (long long) n, (long long) k, (long long) e);
            failures++;
            continue;
        }

        // ggml's own accounting for a contiguous [k, n, experts] tensor of this type -- the third source, and the
        // one neither quactlize's registry nor its units_bytes can influence.
        const int64_t nbytes = (int64_t) ggml_row_size((ggml_type) qtype, k) * n * e;
        const int64_t total  = low + high + units;

        const bool ok = total == nbytes;
        if (!ok) {
            failures++;
        }
        printf("      n=%-6lld k=%-5lld e=%-4lld low=%-12lld high=%-11lld units=%-10lld sum=%-12lld "
               "ggml=%-12lld %s\n",
               (long long) n, (long long) k, (long long) e, (long long) low, (long long) high,
               (long long) units, (long long) total, (long long) nbytes, ok ? "ok" : "FAIL");
    }
    return failures;
}

// Exercises ggml_quactlize_convert_verified end to end against the stub's packing: threading, the byte-exact
// round trip, and -- with QZ_STUB_INTERLEAVE -- the retry that tells a bad split from a bad library. That retry is
// otherwise code nobody runs until the day it matters.
static int run_convert(const qz_case & c) {
    int failures = 0;

    quactlize_ppu_placed_arrangement_v2 a;
    memset(&a, 0, sizeof(a));
    if (!ggml_quactlize_arrangement_for(c.qtype, &a)) {
        printf("    no arrangement for %s -- cannot exercise conversion\n", ggml_type_name((ggml_type) c.qtype));
        return 1;
    }

    const int64_t n = 4, k = 512, experts = 8;
    const int64_t nbytes = (int64_t) ggml_row_size((ggml_type) c.qtype, k) * n * experts;

    int64_t low_b = 0, high_b = 0, units_b = 0;
    if (!ggml_quactlize_plane_sizes(c.qtype, n, k, experts, &a, &low_b, &high_b, &units_b)) {
        printf("    no plane sizes\n");
        return 1;
    }
    if (low_b + high_b + units_b != nbytes) {
        printf("    planes do not add up: %lld vs %lld\n", (long long)(low_b+high_b+units_b), (long long) nbytes);
        return 1;
    }

    std::vector<unsigned char> blocks((size_t) nbytes), recovered((size_t) nbytes);
    std::vector<unsigned char> low((size_t) low_b), high((size_t) (high_b ? high_b : 1)), units((size_t) units_b);
    // Deterministic but not uniform: a constant fill would survive a wrong offset.
    unsigned int seed = 12345;
    for (auto & b : blocks) { seed = seed*1103515245u + 12345u; b = (unsigned char) (seed >> 16); }

    int threads_used = -1;
    const int rc = ggml_quactlize_convert_verified(
        c.qtype, blocks.data(), low.data(), high_b ? high.data() : nullptr, units.data(), recovered.data(),
        nbytes, n, k, experts, &a, low_b, high_b, units_b, &threads_used);

    const bool ok_rc = (rc == 0) == c.want_available;
    if (!ok_rc) failures++;
    printf("    %-34s rc=%-4d want=%s %s\n", "convert_verified", rc, c.want_available ? "0" : "!=0",
           ok_rc ? "ok" : "FAIL");

    const bool threaded = threads_used > 1;
    const bool ok_thr = threaded == c.want_arrangement;
    if (!ok_thr) failures++;
    printf("    %-34s threads=%-3d want=%s %s\n", "threads_used", threads_used,
           c.want_arrangement ? ">1" : "==1", ok_thr ? "ok" : "FAIL");

    return failures;
}

static int run_one(const qz_case & c) {
    int failures = 0;

    auto check = [&](const char * what, bool got, bool want) {
        if (got != want) {
            failures++;
        }
        printf("    %-34s got=%-5s want=%-5s %s\n", what, got ? "true" : "false", want ? "true" : "false",
               got == want ? "ok" : "FAIL");
    };

    check("available",            ggml_quactlize_available(c.qtype),            c.want_available);

    quactlize_ppu_placed_arrangement_v2 a;
    memset(&a, 0, sizeof(a));
    const bool got_arr = ggml_quactlize_arrangement_for(c.qtype, &a);
    check("arrangement_for",      got_arr,                                      c.want_arrangement);

    check("conversion_available", ggml_quactlize_conversion_available(c.qtype), c.want_conversion);

    if (got_arr) {
        printf("      layout=%d bits=%d high=%d gs=%d transport=%d atk=%d\n",
               (int) a.layout, (int) a.bits, (int) a.high_bits, (int) a.group_size,
               (int) a.transport_tile_k, (int) a.artifact_tile_k);
        if (a.artifact_tile_k != 0) {
            printf("      FAIL: artifact_tile_k must be 0 for K-pack\n");
            failures++;
        }
        // Only for a library that is fully usable. Byte neutrality is a claim about an artifact that can actually
        // be produced, and the units plane's size comes from the conversion side -- so a case that deliberately
        // breaks conversion has no units size to check, which is the correct outcome and not a finding.
        if (c.want_conversion) {
            const int mismatches = check_byte_neutrality(c.qtype, a);
            const bool neutral   = mismatches == 0;
            if (neutral != c.want_byte_neutral) {
                printf("    %-34s got=%-5s want=%-5s FAIL\n", "byte_neutral",
                       neutral ? "true" : "false", c.want_byte_neutral ? "true" : "false");
                failures++;
            } else {
                printf("    %-34s got=%-5s want=%-5s ok\n", "byte_neutral",
                       neutral ? "true" : "false", c.want_byte_neutral ? "true" : "false");
            }
        }
    }

    return failures;
}

int main(int argc, char ** argv) {
    if (argc >= 3 && strcmp(argv[1], "--case") == 0) {
        for (size_t i = 0; i < g_ncases; ++i) {
            if (strcmp(g_cases[i].name, argv[2]) == 0) {
                const int f = g_cases[i].kind == QZ_CONVERT ? run_convert(g_cases[i]) : run_one(g_cases[i]);
                return f == 0 ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown case '%s'\n", argv[2]);
        return 2;
    }

    const char * stub_dir = getenv("QZ_STUB_DIR");
    if (stub_dir == nullptr) {
        printf("test-quactlize-loader: QZ_STUB_DIR is not set -- skipping\n");
        return 0;
    }

    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        fprintf(stderr, "cannot find my own path\n");
        return 2;
    }
    self[n] = '\0';

    int failures = 0;
    for (size_t i = 0; i < g_ncases; ++i) {
        printf("  [%2zu/%2zu] %-18s %s\n", i + 1, g_ncases, g_cases[i].name, g_cases[i].why);

        // The loader path is set per case rather than inherited, so "nothing is installed" is a case like any
        // other instead of a separate script nobody runs.
        std::string cmd = "env LD_LIBRARY_PATH=";
        cmd += g_cases[i].hide_bundle ? "" : stub_dir;
        cmd += " ";
        cmd += g_cases[i].env;
        cmd += " '";
        cmd += self;
        cmd += "' --case ";
        cmd += g_cases[i].name;

        const int rc = system(cmd.c_str());
        if (rc != 0) {
            failures++;
            printf("    -> case FAILED (rc=%d)\n", rc);
        }
    }

    printf("test-quactlize-loader: %zu cases, %d failed\n", g_ncases, failures);
    return failures == 0 ? 0 : 1;
}
