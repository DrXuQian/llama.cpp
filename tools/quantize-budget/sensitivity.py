#!/usr/bin/env python3
"""sensitivity: measure how much loss a unit of probe error costs, per tensor category, and turn it into weights.

WHY THIS EXISTS. The probe's error is E||(W-Q)x||^2 in each tensor's OWN output units. Those units are not
comparable across categories: a perturbation of attn_q moves attention logits (then a softmax); one of ffn_down or
attn_output lands directly in the residual stream; one of token_embd in a tied model perturbs the LM head. Adding
raw SSE across categories therefore misallocates -- measured on Qwen2.5-0.5B, an SSE-optimal plan at Q4_K_M's size
was WORSE than llama.cpp's hand-tuned mixture (dPPL 0.99 vs 0.47, max KL 9.6 vs 2.6).

WHAT IT MEASURES. For each category c and one or two "probe types" (a low type such as Q4_0 or Q3_K), quantize ONLY
that category and leave everything else f16; run llama-perplexity with KL-divergence against the f16 logits; read
the mean KL. The probe table gives the summed SSE of exactly that configuration, so
        rate_c = KL_c / SSE_c
is the exchange rate from probe error to loss for that category (first order, linear in SSE). The planner's
objective becomes sum_c rate_c * SSE, which is a first-order estimate of total KL -- the same idea as HAWQ's
Hessian-trace weighting, measured end to end instead of derived from curvature. Cost: one perplexity run per
(category, probe type).

    sensitivity.py --probe probe.json --model f16.gguf --imatrix im.gguf --f16-logits f16.kld --eval text.txt
                   --bin build/bin --types Q4_0 [--categories attn_q,...] -o weights.json
    quantize_budget.py plan ... --weights-json weights.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_budget import load_probe, write_recipe  # noqa: E402


def mean_kld(log_path: str):
    txt = open(log_path, errors="replace").read()
    m = re.search(r"Mean\s+KLD\s*:\s*([0-9.eE+-]+)", txt)
    if not m:
        m = re.search(r"KLD.*?:\s*([0-9.eE+-]+)", txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--model", required=True, help="f16/bf16 source GGUF")
    ap.add_argument("--imatrix")
    ap.add_argument("--f16-logits", required=True, help="logits saved by llama-perplexity --kl-divergence-base on the source")
    ap.add_argument("--eval", required=True, help="evaluation text (same as used for the f16 logits)")
    ap.add_argument("--bin", required=True, help="llama.cpp build dir containing bin/")
    ap.add_argument("--types", default="Q4_0", help="comma-separated probe types; each category is quantized to the first one it can take")
    ap.add_argument("--categories", help="comma-separated subset (default: every category in the probe)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("-o", "--out", default="weights.json")
    a = ap.parse_args()

    p, tensors = load_probe(a.probe)
    cats = sorted({t.category for t in tensors})
    if a.categories:
        cats = [c for c in a.categories.split(",") if c in cats]
    probe_types = [x.strip().lower() for x in a.types.split(",")]
    work = a.workdir or os.path.join(os.path.dirname(os.path.abspath(a.probe)), "sens")
    os.makedirs(work, exist_ok=True)

    results = {}
    for cat in cats:
        members = [t for t in tensors if t.category == cat]
        # pick, per tensor, the first probe type it has a candidate for (k-quants need ne0 % 256 == 0)
        choice, sse = [], 0.0
        for t in members:
            c = None
            for pt in probe_types:
                c = next((c for c in t.cands if c[0].lower() == pt), None)
                if c: break
            if c is None:
                c = min(t.cands, key=lambda c: c[1])   # smallest valid candidate as a last resort
            choice.append(c); sse += c[2]
        if sse <= 0:
            print(f"{cat:16s} skipped (no error at the probe type)"); continue
        recipe = os.path.join(work, f"{cat}.recipe")
        # everything NOT in the category stays at its source type: pin every other tensor to f16 explicitly, since
        # llama-quantize would otherwise quantize them with the base ftype
        others = [t for t in tensors if t.category != cat]
        write_recipe(recipe, members + others, choice + [(t.src_type, t.bytes_src, 0.0, 0.0, 0.0) for t in others],
                     {"category": cat, "probe_types": probe_types})
        gguf = os.path.join(work, f"{cat}.gguf")
        if not os.path.exists(gguf):
            cmd = [os.path.join(a.bin, "bin", "llama-quantize")]
            if a.imatrix: cmd += ["--imatrix", a.imatrix]
            cmd += ["--tensor-type-file", recipe, a.model, gguf, probe_types[0].upper(), str(a.threads)]
            with open(os.path.join(work, f"{cat}.quantize.log"), "w") as lf:
                if subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT) != 0:
                    print(f"{cat:16s} llama-quantize failed (see {lf.name})"); continue
        log = os.path.join(work, f"{cat}.ppl.log")
        if not os.path.exists(log):
            cmd = [os.path.join(a.bin, "bin", "llama-perplexity"), "-m", gguf, "-f", a.eval, "-c", str(a.ctx), "--chunks", str(a.chunks),
                   "-t", str(a.threads), "--kl-divergence-base", a.f16_logits, "--kl-divergence"]
            with open(log, "w") as lf:
                subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        # each probe GGUF is mostly f16 (~the source's size); keep the logs, not the files
        if os.path.exists(gguf) and os.path.exists(log):
            os.unlink(gguf)
        kl = mean_kld(log)
        if kl is None:
            print(f"{cat:16s} no KL in {log}"); continue
        rate = kl / sse
        results[cat] = {"n_tensors": len(members), "types": sorted({c[0] for c in choice}), "sse": sse, "mean_kld": kl, "rate": rate}
        print(f"{cat:16s} {len(members):3d} tensors  types={','.join(sorted({c[0] for c in choice})):12s}  SSE={sse:10.4e}  meanKL={kl:.5f}  rate={rate:.4e}")

    if not results:
        sys.exit("nothing measured")
    # normalise so the median category has weight 1: the absolute scale does not matter to the allocator
    rates = sorted(r["rate"] for r in results.values())
    med = rates[len(rates) // 2]
    weights = {c: r["rate"] / med for c, r in results.items()}
    with open(a.out, "w") as f:
        json.dump({"weights": weights, "measurements": results, "probe_types": probe_types,
                   "note": "weight_c = (meanKL_c / SSE_c) / median; objective sum_c weight_c * SSE_c ~ first-order KL"}, f, indent=1)
    print("\nweights (median category = 1):")
    for c, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {c:16s} {w:8.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
