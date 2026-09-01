// The llama.cpp half of the quactlize K-pack loader, against a stub library.
//
// WHY THIS EXISTS AND WHY IT RUNS OFF-BOX. The guards under test decide whether a tensor's GGUF bytes get replaced
// by a K-pack artifact -- after which there is no second copy and no fallback. They are therefore the guards most
// worth having a negative control for, and they are also the only part of the path that needs neither a PPU device
// nor a real kernel library: identity, the format-registry cross-check, and capability-vs-presence. The stub beside
// this file plays a library that lies in one specific way per case.
//
// The loader arms itself once per process (pthread_once), so one case per process: with no --case the binary is the
// driver and re-execs itself, once per row of the table below, with that row's environment.

#include "quactlize-lib.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

struct qz_case {
    const char * name;
    const char * env;              // what makes the stub lie, "" for the honest library
    int          qtype;
    bool         want_available;
    bool         want_arrangement;
    bool         want_conversion;
    const char * why;
};

// Two positive controls and five negatives. Every negative must make at least one column flip to false: a guard that
// cannot be made to fire is not a guard.
static const qz_case g_cases[] = {
    { "honest-q4k",        "",                            GGML_TYPE_Q4_K, true,  true,  true,
      "fmt0 present, identity 0, arrangement matches the registry" },
    { "absent-format",     "",                            GGML_TYPE_Q5_K, false, false, false,
      "no libquactlize_ppu_fmt1.so on the path -- unavailable, not a crash" },
    { "wrong-identity",    "QZ_STUB_FMT=3",               GGML_TYPE_Q4_K, false, false, false,
      "the file named fmt0 reports format 3: a reshuffled bundle must not arm" },
    { "default-library",   "QZ_STUB_NO_IDENTITY=1",       GGML_TYPE_Q4_K, false, false, false,
      "the default/ScaleFirst build reports -1 and is not a K-pack library" },
    { "registry-mismatch", "QZ_STUB_BAD_GS=1",            GGML_TYPE_Q4_K, true,  false, false,
      "group_size off by one against ppu_format_config.inc" },
    { "xplane-descriptor", "QZ_STUB_BAD_ATK=1",           GGML_TYPE_Q4_K, true,  false, false,
      "artifact_tile_k != 0: an Xplane descriptor arriving by the K-pack door" },
    { "no-conversion",     "QZ_STUB_NO_CONVERSION=1",     GGML_TYPE_Q4_K, true,  true,  false,
      "the conversion entries are exported but cannot answer for a 256-code superblock" },
};

static const size_t g_ncases = sizeof(g_cases) / sizeof(g_cases[0]);

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
    if (got_arr) {
        printf("      layout=%d bits=%d high=%d gs=%d transport=%d atk=%d\n",
               (int) a.layout, (int) a.bits, (int) a.high_bits, (int) a.group_size,
               (int) a.transport_tile_k, (int) a.artifact_tile_k);
        // A K-pack artifact has no artifact-TileK axis, in every case that gets this far.
        if (a.artifact_tile_k != 0) {
            printf("      FAIL: artifact_tile_k must be 0 for K-pack\n");
            failures++;
        }
    }

    check("conversion_available", ggml_quactlize_conversion_available(c.qtype), c.want_conversion);

    return failures;
}

int main(int argc, char ** argv) {
    // --case <name>: one case, in this process, with the environment the driver already set.
    if (argc >= 3 && strcmp(argv[1], "--case") == 0) {
        for (size_t i = 0; i < g_ncases; ++i) {
            if (strcmp(g_cases[i].name, argv[2]) == 0) {
                return run_one(g_cases[i]) == 0 ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown case '%s'\n", argv[2]);
        return 2;
    }

    if (!ggml_quactlize_available(GGML_TYPE_Q4_K) && getenv("QZ_STUB_DIR") == nullptr) {
        printf("test-quactlize-loader: QZ_STUB_DIR is not set and no stub is on the loader path -- skipping\n");
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
        printf("  [%zu/%zu] %-18s %s\n", i + 1, g_ncases, g_cases[i].name, g_cases[i].why);

        std::string cmd = "env ";
        cmd += g_cases[i].env;
        cmd += " QZ_STUB_QTYPE=";
        cmd += std::to_string(GGML_TYPE_Q4_K);   // the stub always plays the Q4_K library
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
