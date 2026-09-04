// llama-quant-probe: per-tensor, per-type quantization error, measured with ggml's own quantizers.
//
// WHAT IT ANSWERS. For every weight tensor of a GGUF and every candidate quantization type: how many bytes would it
// take, and how much imatrix-weighted error would it introduce? That table is the whole input a budget allocator
// needs, and nothing in it is a guess: the numbers come from running the exact quantizer llama-quantize would run,
// on a uniform sample of the tensor's rows, with the same imatrix column weights.
//
// THE ERROR METRIC. With an importance matrix m_j = E[x_j^2] over calibration tokens, the quantity
//     sse = sum_i sum_j m_j (w_ij - q_ij)^2
// is the expected squared perturbation of the tensor's output, E||(W - Q) x||^2, under a diagonal-Hessian model of
// the input -- the same objective the imatrix-aware quantizers minimise inside ggml. It is reported next to the
// tensor's own output energy sum m_j w_ij^2, so the allocator can work with either the absolute perturbation (which
// is comparable across tensors that write into the same residual stream) or the relative one (an SNR). Without an
// imatrix, m_j = 1 and the metric degrades to plain weight-space MSE, which is what llama-quantize optimises then.
//
// WHAT IT DOES NOT ANSWER. How much a given perturbation costs in final loss -- HAWQ-style second-order sensitivity
// with respect to the loss, or a measured KL divergence -- is a separate measurement layered on top of this table,
// not folded into it.
//
//     llama-quant-probe -m model.gguf [--imatrix im.gguf] [--types Q2_K,Q3_K,...] [--rows 256] [-t threads] -o probe.json

#include "ggml.h"
#include "gguf.h"
#include "imatrix-loader.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct args_t {
    std::string model, imatrix, out = "probe.json";
    std::vector<std::string> types;
    int rows = 256;
    int threads = 0;
};

const char * DEFAULT_TYPES[] = {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
    "Q4_0", "Q5_0", "F16",
};

bool parse_args(int argc, char ** argv, args_t & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&](std::string & dst) { if (i + 1 >= argc) { return false; } dst = argv[++i]; return true; };
        std::string v;
        if (s == "-m" || s == "--model") { if (!next(a.model)) return false; }
        else if (s == "--imatrix") { if (!next(a.imatrix)) return false; }
        else if (s == "-o" || s == "--out") { if (!next(a.out)) return false; }
        else if (s == "--rows") { if (!next(v)) return false; a.rows = atoi(v.c_str()); }
        else if (s == "-t" || s == "--threads") { if (!next(v)) return false; a.threads = atoi(v.c_str()); }
        else if (s == "--types") {
            if (!next(v)) return false;
            size_t p = 0;
            while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); if (q > p) a.types.push_back(v.substr(p, q - p)); p = q + 1; }
        }
        else { fprintf(stderr, "unknown argument %s\n", s.c_str()); return false; }
    }
    if (a.model.empty()) { fprintf(stderr, "usage: llama-quant-probe -m model.gguf [--imatrix im.gguf] [--types A,B,...] [--rows N] [-t N] -o probe.json\n"); return false; }
    if (a.types.empty()) { for (auto t : DEFAULT_TYPES) a.types.push_back(t); }
    if (a.threads <= 0) { a.threads = (int) std::max(1u, std::thread::hardware_concurrency()); }
    return true;
}

ggml_type type_from_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcasecmp(n, name.c_str()) == 0) { return (ggml_type) t; }
    }
    return GGML_TYPE_COUNT;
}

// Tensor roles by name, the way llama.cpp's quantizer and its readers see them.
struct role_t { std::string category; int layer = -1; bool weight = false; bool router = false; };

role_t classify(const std::string & name, int ndims) {
    role_t r;
    if (name.rfind(".weight") != name.size() - 7 || ndims < 2) { return r; }
    r.weight = true;
    if (name.rfind("blk.", 0) == 0) {
        r.layer = atoi(name.c_str() + 4);
    }
    auto has = [&](const char * s) { return name.find(s) != std::string::npos; };
    if (name == "token_embd.weight")        r.category = "token_embd";
    else if (name == "output.weight")       r.category = "output";
    else if (has("ffn_gate_inp"))           { r.category = "router"; r.router = true; }
    else if (has("ffn_gate_exps"))          r.category = "ffn_gate_exps";
    else if (has("ffn_up_exps"))            r.category = "ffn_up_exps";
    else if (has("ffn_down_exps"))          r.category = "ffn_down_exps";
    else if (has("ffn_gate_shexp"))         r.category = "ffn_gate_shexp";
    else if (has("ffn_up_shexp"))           r.category = "ffn_up_shexp";
    else if (has("ffn_down_shexp"))         r.category = "ffn_down_shexp";
    else if (has("ffn_gate"))               r.category = "ffn_gate";
    else if (has("ffn_up"))                 r.category = "ffn_up";
    else if (has("ffn_down"))               r.category = "ffn_down";
    else if (has("attn_q"))                 r.category = "attn_q";
    else if (has("attn_k"))                 r.category = "attn_k";
    else if (has("attn_v"))                 r.category = "attn_v";
    else if (has("attn_output") || has("attn_o.")) r.category = "attn_output";
    else if (has("attn_qkv"))               r.category = "attn_qkv";
    else                                    r.category = "other";
    return r;
}

std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char) c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += (char) c;
    }
    return o;
}

struct candidate_t {
    ggml_type type;
    size_t    bytes;       // for the whole tensor
    double    sse;         // imatrix-weighted SSE, scaled to the whole tensor
    double    energy;      // imatrix-weighted sum of w^2, scaled to the whole tensor
    double    mse;         // plain, per element
    bool      ok;
};

struct tensor_t {
    std::string name;
    role_t      role;
    ggml_type   src_type;
    int64_t     ne[4];
    int         ndims;
    size_t      bytes_src;
    const uint8_t * data;
    const float * imatrix;     // n_per_row * n_expert, or null
    int64_t     imatrix_n_expert;
    std::vector<candidate_t> cands;
};

} // namespace

int main(int argc, char ** argv) {
    args_t a;
    if (!parse_args(argc, argv, a)) { return 1; }

    std::vector<ggml_type> types;
    for (const auto & n : a.types) {
        const ggml_type t = type_from_name(n);
        if (t == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", n.c_str()); return 1; }
        types.push_back(t);
    }

    // ---- the model, mapped ----
    const int fd = open(a.model.c_str(), O_RDONLY);
    if (fd < 0) { perror(a.model.c_str()); return 1; }
    struct stat st{}; fstat(fd, &st);
    const uint8_t * base = (const uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta_ctx };
    gguf_context * g = gguf_init_from_file(a.model.c_str(), gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", a.model.c_str()); return 1; }
    const size_t data_off = gguf_get_data_offset(g);

    // ---- the imatrix ----
    std::map<std::string, std::vector<float>> im;
    if (!a.imatrix.empty()) {
        common_imatrix loaded;
        if (!common_imatrix_load(a.imatrix, loaded)) { fprintf(stderr, "cannot load imatrix %s\n", a.imatrix.c_str()); return 1; }
        for (const auto & [name, e] : loaded.entries) {
            auto & v = im[name];
            v.resize(e.sums.size());
            if (!loaded.is_legacy) {
                const int64_t nc = (int64_t) e.counts.size(), ne0 = (int64_t) e.sums.size() / std::max<int64_t>(1, nc);
                for (int64_t j = 0; j < nc; ++j) {
                    const float c = (float) e.counts[j];
                    for (int64_t i = 0; i < ne0; ++i) { v[j*ne0 + i] = c > 0 ? e.sums[j*ne0 + i] / c : 1.0f; }
                }
            } else {
                const int64_t ncall = e.counts.empty() ? 0 : e.counts[0];
                for (size_t i = 0; i < e.sums.size(); ++i) { v[i] = ncall > 0 ? e.sums[i] / ncall : e.sums[i]; }
            }
        }
        fprintf(stderr, "imatrix: %zu entries from %s\n", im.size(), a.imatrix.c_str());
    }

    // ---- inventory ----
    std::vector<tensor_t> tensors;
    size_t bytes_other = 0;
    for (int i = 0; i < (int) gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        tensor_t T;
        T.name = name; T.ndims = ggml_n_dims(t); T.src_type = t->type;
        for (int d = 0; d < 4; ++d) T.ne[d] = t->ne[d];
        T.bytes_src = ggml_nbytes(t);
        T.data = base + data_off + gguf_get_tensor_offset(g, i);
        T.role = classify(T.name, T.ndims);
        const bool quantizable = T.role.weight && !T.role.router &&
            (T.src_type == GGML_TYPE_F16 || T.src_type == GGML_TYPE_BF16 || T.src_type == GGML_TYPE_F32);
        if (!quantizable) { bytes_other += T.bytes_src; continue; }
        auto it = im.find(T.name);
        T.imatrix = it != im.end() ? it->second.data() : nullptr;
        T.imatrix_n_expert = it != im.end() ? (int64_t) (it->second.size() / T.ne[0]) : 0;
        if (T.imatrix && (int64_t) it->second.size() != T.ne[0] * std::max<int64_t>(1, T.ne[2])) {
            fprintf(stderr, "warning: imatrix size for %s does not match the tensor -- ignoring it for this tensor\n", name);
            T.imatrix = nullptr;
        }
        tensors.push_back(T);
    }
    fprintf(stderr, "%zu quantizable tensors, %.1f MiB of others kept as is\n", tensors.size(), bytes_other / 1048576.0);

    // ---- the measurement ----
    std::atomic<size_t> next{0};
    std::mutex log_mutex;
    auto worker = [&]() {
        std::vector<float> rows, deq;
        std::vector<uint8_t> qbuf;
        for (size_t ti; (ti = next.fetch_add(1)) < tensors.size();) {
            tensor_t & T = tensors[ti];
            const int64_t ncols = T.ne[0], nrows_mat = T.ne[1], nexp = std::max<int64_t>(1, T.ne[2]);
            const int64_t nrows_total = nrows_mat * nexp;
            const int64_t R = std::min<int64_t>(a.rows, nrows_total);
            const auto * src_traits = ggml_get_type_traits(T.src_type);
            const size_t src_row = ggml_row_size(T.src_type, ncols);
            const double scale = (double) nrows_total / (double) R;

            // Uniformly spaced rows across experts and rows: index r -> (expert, row).
            std::vector<std::pair<int64_t,int64_t>> picks;
            for (int64_t s = 0; s < R; ++s) {
                const int64_t flat = (nrows_total * s) / R + (nrows_total / R) / 2;
                picks.emplace_back(flat / nrows_mat, flat % nrows_mat);
            }
            std::sort(picks.begin(), picks.end());
            rows.resize((size_t) R * ncols);
            for (int64_t s = 0; s < R; ++s) {
                const uint8_t * p = T.data + (picks[s].first * nrows_mat + picks[s].second) * src_row;
                if (T.src_type == GGML_TYPE_F32) memcpy(&rows[s * ncols], p, ncols * sizeof(float));
                else src_traits->to_float(p, &rows[s * ncols], ncols);
            }

            for (const ggml_type qt : types) {
                candidate_t c; c.type = qt; c.ok = false; c.sse = c.energy = c.mse = 0;
                c.bytes = ggml_row_size(qt, ncols) * nrows_total;
                if (ncols % ggml_blck_size(qt) != 0) { T.cands.push_back(c); continue; }
                if (ggml_quantize_requires_imatrix(qt) && !T.imatrix) { T.cands.push_back(c); continue; }
                if (qt == T.src_type) { c.ok = true; T.cands.push_back(c); continue; }
                const auto * qtraits = ggml_get_type_traits(qt);
                if (!qtraits->to_float) { T.cands.push_back(c); continue; }
                qbuf.resize(ggml_row_size(qt, ncols) * R + 64);
                deq.resize((size_t) R * ncols);
                // quantize in runs of rows that share an expert (and therefore an imatrix slice)
                int64_t s = 0;
                while (s < R) {
                    int64_t e = s; const int64_t ex = picks[s].first;
                    while (e < R && picks[e].first == ex) ++e;
                    const float * imx = T.imatrix ? T.imatrix + (T.imatrix_n_expert > 1 ? ex : 0) * ncols : nullptr;
                    uint8_t * q = qbuf.data() + ggml_row_size(qt, ncols) * s;
                    ggml_quantize_chunk(qt, &rows[s * ncols], q, 0, e - s, ncols, imx);
                    qtraits->to_float(q, &deq[s * ncols], (e - s) * ncols);
                    s = e;
                }
                double sse = 0, energy = 0, plain = 0;
                for (int64_t r = 0; r < R; ++r) {
                    const float * imx = T.imatrix ? T.imatrix + (T.imatrix_n_expert > 1 ? picks[r].first : 0) * ncols : nullptr;
                    const float * w = &rows[r * ncols]; const float * q = &deq[r * ncols];
                    for (int64_t j = 0; j < ncols; ++j) {
                        const double d = (double) w[j] - (double) q[j];
                        const double m = imx ? (double) imx[j] : 1.0;
                        sse += m * d * d; energy += m * (double) w[j] * (double) w[j]; plain += d * d;
                    }
                }
                c.ok = true; c.sse = sse * scale; c.energy = energy * scale; c.mse = plain / ((double) R * ncols);
                T.cands.push_back(c);
            }
            std::lock_guard<std::mutex> lk(log_mutex);
            fprintf(stderr, "  %-40s %-14s [%5" PRId64 " x %5" PRId64 " x %3" PRId64 "]  %zu types\n",
                    T.name.c_str(), T.role.category.c_str(), T.ne[0], T.ne[1], nexp, T.cands.size());
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < a.threads; ++i) pool.emplace_back(worker);
    for (auto & th : pool) th.join();

    // ---- the table ----
    FILE * f = fopen(a.out.c_str(), "w");
    if (!f) { perror(a.out.c_str()); return 1; }
    fprintf(f, "{\n  \"model\": \"%s\",\n  \"imatrix\": %s,\n  \"rows_sampled\": %d,\n  \"bytes_other\": %zu,\n  \"types\": [",
            json_escape(a.model).c_str(), a.imatrix.empty() ? "false" : "true", a.rows, bytes_other);
    for (size_t i = 0; i < types.size(); ++i) fprintf(f, "%s\"%s\"", i ? ", " : "", ggml_type_name(types[i]));
    fprintf(f, "],\n  \"tensors\": [\n");
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & T = tensors[i];
        fprintf(f, "    {\"name\": \"%s\", \"category\": \"%s\", \"layer\": %d, \"src_type\": \"%s\", \"shape\": [%" PRId64 ", %" PRId64 ", %" PRId64 "], "
                   "\"n_elements\": %" PRId64 ", \"bytes_src\": %zu, \"imatrix\": %s, \"candidates\": [",
                json_escape(T.name).c_str(), T.role.category.c_str(), T.role.layer, ggml_type_name(T.src_type),
                T.ne[0], T.ne[1], std::max<int64_t>(1, T.ne[2]), T.ne[0] * T.ne[1] * std::max<int64_t>(1, T.ne[2]),
                T.bytes_src, T.imatrix ? "true" : "false");
        bool first = true;
        for (const auto & c : T.cands) {
            if (!c.ok) continue;
            fprintf(f, "%s{\"type\": \"%s\", \"bytes\": %zu, \"sse\": %.6e, \"energy\": %.6e, \"mse\": %.6e}",
                    first ? "" : ", ", ggml_type_name(c.type), c.bytes, c.sse, c.energy, c.mse);
            first = false;
        }
        fprintf(f, "]}%s\n", i + 1 < tensors.size() ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    fprintf(stderr, "wrote %s\n", a.out.c_str());
    gguf_free(g); ggml_free(meta_ctx); munmap((void *) base, st.st_size); close(fd);
    return 0;
}
