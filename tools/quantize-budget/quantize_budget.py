#!/usr/bin/env python3
"""quantize-budget: choose a quantization type per tensor under a byte budget, then apply it with llama-quantize.

The pipeline has three parts, and only the middle one lives here:

    llama-quant-probe   measures, per tensor and per candidate type, the bytes and the imatrix-weighted error
                        (ggml's own quantizers on a sample of rows)                              -> probe.json
    quantize_budget.py  turns that table into an allocation under a budget                       -> recipe.txt
    llama-quantize      applies the recipe through --tensor-type-file                            -> model.gguf

WHAT THE ALLOCATION MINIMISES. Each tensor t has candidates c with bytes b_tc and an error e_tc. The plan picks one
candidate per tensor to minimise sum_t w_t * e_tc subject to sum_t b_tc <= budget. e_tc is, by default, the
absolute imatrix-weighted SSE the probe measured -- the expected squared perturbation of that tensor's output --
which is the same objective ggml's imatrix quantizers minimise inside one tensor, extended across tensors. w_t is a
per-category weight (1.0 unless told otherwise) that is the hook for what this metric cannot see: how much a
perturbation in that tensor costs in final loss. The `sensitivity` command exists to measure that.

HOW. Multiple-choice knapsack, solved exactly by dynamic programming over the byte budget in fixed-size units
(default 256 KiB; the leftover under that granularity is spent greedily), with a Lagrangian relaxation available as
a fast approximation and as a cross-check. Both are deterministic.

    quantize_budget.py plan   --probe probe.json --budget 6.5G [--reserve 1G] [--types Q3_K,Q4_K,...]
                              [--keep output=Q6_K] [--weight ffn_down=2.0] [--objective sse|rel] -o recipe.txt
    quantize_budget.py apply  --model f16.gguf --recipe recipe.txt --out model.gguf [--imatrix im.gguf]
                              [--quantize-bin llama-quantize]
    quantize_budget.py pareto --probe probe.json --from 3G --to 8G --steps 11        (the curve, no files written)
    quantize_budget.py import-llama --log dryrun.log --probe probe.json               (llama-quantize's own choice,
                                                                                       for comparison at equal size)
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

try:
    import numpy as np
except ImportError:  # the planner is small enough to stay usable without numpy, just slower
    np = None


# --------------------------------------------------------------------------------------------------------------
# sizes and tables
# --------------------------------------------------------------------------------------------------------------

def parse_bytes(s: str) -> int:
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kmgt]?i?b?)\s*", s, re.I)
    if not m:
        raise ValueError(f"cannot parse size {s!r}")
    v, unit = float(m.group(1)), m.group(2).lower()
    mult = {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10, "m": 1 << 20, "mb": 1 << 20, "mib": 1 << 20,
            "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30, "t": 1 << 40, "tb": 1 << 40, "tib": 1 << 40}
    return int(v * mult[unit])


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TiB"


class Tensor:
    def __init__(self, rec: dict):
        self.name = rec["name"]
        self.category = rec["category"]
        self.layer = rec["layer"]
        self.shape = rec["shape"]
        self.n_elements = rec["n_elements"]
        self.bytes_src = rec["bytes_src"]
        self.src_type = rec["src_type"]
        # candidates: list of (type, bytes, sse, energy, mse)
        self.cands = [(c["type"], int(c["bytes"]), float(c["sse"]), float(c["energy"]), float(c["mse"]))
                      for c in rec["candidates"]]


def load_probe(path: str):
    with open(path) as f:
        p = json.load(f)
    tensors = [Tensor(r) for r in p["tensors"]]
    return p, tensors


def objective_of(t: Tensor, cand, objective: str, weight: float) -> float:
    typ, b, sse, energy, mse = cand
    if objective == "sse":
        e = sse
    elif objective == "rel":
        e = sse / energy if energy > 0 else 0.0
    elif objective == "mse":
        e = mse * t.n_elements
    else:
        raise ValueError(objective)
    return weight * e


def category_weights(spec: list) -> dict:
    w = {}
    for item in spec or []:
        k, v = item.split("=")
        w[k] = float(v)
    return w


def filter_candidates(tensors, types: list, keep: dict, min_type: dict, max_bits: dict):
    """Apply the user's constraints: allowed type set, per-category pins, per-category floors."""
    allowed = set(types) if types else None
    for t in tensors:
        pinned = None
        for pat, typ in keep.items():
            if re.search(pat, t.name) or pat == t.category:
                pinned = typ
        cands = []
        for c in t.cands:
            typ = c[0]
            if pinned is not None:
                if typ.lower() == pinned.lower():
                    cands.append(c)
                continue
            if allowed is not None and typ.upper() not in allowed and typ.lower() not in allowed:
                continue
            cands.append(c)
        if pinned is not None and not cands:
            # the pinned type was not probed (or not valid for this shape): fall back to the source type
            cands = [c for c in t.cands if c[0] == t.src_type] or t.cands[:1]
        if not cands:
            cands = t.cands[:]
        t.cands = sorted(cands, key=lambda c: c[1])


# --------------------------------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------------------------------

def solve_lagrangian(tensors, budget: int, objective: str, weights: dict):
    """argmin_c e_tc + lam * b_tc per tensor, lam found by bisection so that the total fits. Fast, near-optimal
    when error-vs-bytes is convex (it nearly always is), and the cross-check for the DP."""
    def pick(lam):
        total, err, choice = 0, 0.0, []
        for t in tensors:
            w = weights.get(t.category, 1.0)
            best = min(t.cands, key=lambda c: objective_of(t, c, objective, w) + lam * c[1])
            choice.append(best); total += best[1]; err += objective_of(t, best, objective, w)
        return total, err, choice
    lo, hi = 0.0, 1.0
    total, _, _ = pick(hi)
    while total > budget and hi < 1e30:
        hi *= 4; total, _, _ = pick(hi)
    if total > budget:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        total, _, _ = pick(mid)
        if total > budget:
            lo = mid
        else:
            hi = mid
    total, err, choice = pick(hi)
    # spend the slack greedily on the best error reduction per byte
    choice = greedy_fill(tensors, choice, budget, objective, weights)
    return choice


def greedy_fill(tensors, choice, budget, objective, weights):
    choice = list(choice)
    total = sum(c[1] for c in choice)
    improved = True
    while improved:
        improved = False
        best = None
        for i, t in enumerate(tensors):
            w = weights.get(t.category, 1.0)
            cur = choice[i]; cur_e = objective_of(t, cur, objective, w)
            for c in t.cands:
                if c[1] <= cur[1] or total - cur[1] + c[1] > budget:
                    continue
                gain = (cur_e - objective_of(t, c, objective, w)) / max(1, c[1] - cur[1])
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, i, c)
        if best is not None:
            _, i, c = best
            total += c[1] - choice[i][1]; choice[i] = c; improved = True
    return choice


def solve_dp(tensors, budget: int, objective: str, weights: dict, unit: int):
    """Exact multiple-choice knapsack over the budget in `unit`-byte steps (each candidate's size rounded UP, so the
    plan never exceeds the budget), then the sub-unit slack is spent greedily."""
    if np is None:
        return solve_lagrangian(tensors, budget, objective, weights)
    U = budget // unit
    INF = float("inf")
    f = np.full(U + 1, INF); f[0] = 0.0
    back = []   # per tensor: array of chosen candidate index per budget unit
    for t in tensors:
        w = weights.get(t.category, 1.0)
        g = np.full(U + 1, INF); arg = np.full(U + 1, -1, dtype=np.int32)
        for ci, c in enumerate(t.cands):
            s = -(-c[1] // unit)          # ceil
            if s > U:
                continue
            e = objective_of(t, c, objective, w)
            cand = np.full(U + 1, INF)
            cand[s:] = f[:U + 1 - s] + e
            better = cand < g
            g[better] = cand[better]; arg[better] = ci
        f, back = g, back + [arg]
        if not np.isfinite(f).any():
            return None
    b = int(np.argmin(f))
    if not np.isfinite(f[b]):
        return None
    choice = [None] * len(tensors)
    for i in range(len(tensors) - 1, -1, -1):
        ci = int(back[i][b]); c = tensors[i].cands[ci]; choice[i] = c
        b -= -(-c[1] // unit)
    return greedy_fill(tensors, choice, budget, objective, weights)


# --------------------------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------------------------

def summarize(tensors, choice, objective, weights, bytes_other, budget=None):
    total = sum(c[1] for c in choice) + bytes_other
    n_el = sum(t.n_elements for t in tensors)
    err = sum(objective_of(t, c, objective, weights.get(t.category, 1.0)) for t, c in zip(tensors, choice))
    by_cat = {}
    for t, c in zip(tensors, choice):
        d = by_cat.setdefault(t.category, {})
        d[c[0]] = d.get(c[0], 0) + 1
    hist = {}
    for c in choice:
        hist[c[0]] = hist.get(c[0], 0) + 1
    return {
        "bytes_total": total, "bytes_weights": total - bytes_other, "bytes_other": bytes_other,
        "budget": budget, "bpw": 8.0 * (total - bytes_other) / n_el if n_el else 0.0,
        "objective": objective, "error": err, "types": hist, "by_category": by_cat,
    }


def print_summary(s):
    print(f"  total {fmt_bytes(s['bytes_total'])}  (weights {fmt_bytes(s['bytes_weights'])} + other {fmt_bytes(s['bytes_other'])})"
          + (f"  budget {fmt_bytes(s['budget'])}" if s["budget"] else ""))
    print(f"  bits/weight {s['bpw']:.3f}   objective[{s['objective']}] {s['error']:.6e}")
    print("  types: " + ", ".join(f"{k}:{v}" for k, v in sorted(s["types"].items(), key=lambda kv: -kv[1])))
    for cat, d in sorted(s["by_category"].items()):
        print(f"    {cat:16s} " + ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])))


def write_recipe(path, tensors, choice, header: dict):
    with open(path, "w") as f:
        f.write("# quantize-budget recipe: one anchored pattern per tensor, for llama-quantize --tensor-type-file\n")
        for k, v in header.items():
            f.write(f"# {k}: {v}\n")
        for t, c in zip(tensors, choice):
            f.write(f"^{re.escape(t.name)}$={c[0]}\n")


def read_recipe(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pat, typ = line.rsplit("=", 1)
            out[pat] = typ
    return out


# --------------------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------------------

def cmd_plan(a):
    p, tensors = load_probe(a.probe)
    weights = category_weights(a.weight)
    keep = dict(item.split("=", 1) for item in (a.keep or []))
    types = [t.upper() for t in a.types.split(",")] if a.types else None
    filter_candidates(tensors, types, keep, {}, {})
    budget = parse_bytes(a.budget) - parse_bytes(a.reserve) - p["bytes_other"]
    if budget <= 0:
        sys.exit(f"budget leaves nothing for weights after reserve and {fmt_bytes(p['bytes_other'])} of unquantized tensors")
    solver = solve_dp if a.solver == "dp" else solve_lagrangian
    choice = solver(tensors, budget, a.objective, weights, parse_bytes(a.unit)) if a.solver == "dp" else solver(tensors, budget, a.objective, weights)
    if choice is None:
        smallest = sum(min(c[1] for c in t.cands) for t in tensors)
        sys.exit(f"infeasible: even the smallest candidates need {fmt_bytes(smallest + p['bytes_other'])} > budget")
    s = summarize(tensors, choice, a.objective, weights, p["bytes_other"], parse_bytes(a.budget) - parse_bytes(a.reserve))
    print(f"plan for {p['model']}  (solver {a.solver}, imatrix {'yes' if p['imatrix'] else 'NO'})")
    print_summary(s)
    if a.solver == "dp" and a.cross_check:
        alt = solve_lagrangian(tensors, budget, a.objective, weights)
        if alt is not None:
            s2 = summarize(tensors, alt, a.objective, weights, p["bytes_other"])
            print(f"  cross-check: lagrangian objective {s2['error']:.6e} at {fmt_bytes(s2['bytes_total'])}"
                  f"  (dp is {'better' if s['error'] <= s2['error'] else 'WORSE -- report this'})")
    write_recipe(a.out, tensors, choice, {"model": p["model"], "budget": a.budget, "objective": a.objective,
                                            "bytes_total": s["bytes_total"], "bpw": f"{s['bpw']:.3f}"})
    if a.report:
        with open(a.report, "w") as f:
            json.dump({"summary": s, "plan": [{"name": t.name, "category": t.category, "layer": t.layer,
                                                "type": c[0], "bytes": c[1], "sse": c[2], "energy": c[3]}
                                               for t, c in zip(tensors, choice)]}, f, indent=1)
    print(f"recipe: {a.out}")


def cmd_pareto(a):
    p, tensors = load_probe(a.probe)
    weights = category_weights(a.weight)
    types = [t.upper() for t in a.types.split(",")] if a.types else None
    filter_candidates(tensors, types, {}, {}, {})
    lo, hi = parse_bytes(a.frm), parse_bytes(a.to)
    print(f"{'budget':>12s} {'total':>12s} {'bpw':>7s} {'objective':>14s}  types")
    for i in range(a.steps):
        budget = lo + (hi - lo) * i / max(1, a.steps - 1)
        choice = solve_lagrangian(tensors, int(budget) - p["bytes_other"], a.objective, weights)
        if choice is None:
            print(f"{fmt_bytes(budget):>12s}  infeasible"); continue
        s = summarize(tensors, choice, a.objective, weights, p["bytes_other"], int(budget))
        print(f"{fmt_bytes(budget):>12s} {fmt_bytes(s['bytes_total']):>12s} {s['bpw']:7.3f} {s['error']:14.6e}  "
              + ", ".join(f"{k}:{v}" for k, v in sorted(s['types'].items(), key=lambda kv: -kv[1])[:5]))


def cmd_import_llama(a):
    """Read the per-tensor types llama-quantize printed (a --dry-run or a real run) and score that plan with the
    probe table, so llama.cpp's own mixture can be compared with a budget plan at the same size."""
    p, tensors = load_probe(a.probe)
    by_name = {t.name: t for t in tensors}
    pat = re.compile(r"^\[\s*\d+/\s*\d+\]\s+(\S+)\s+-\s+\[.*?\],\s+type\s*=\s*(\w+),\s+(?:converting to|size\s*=)\s*(\w+)?")
    choice, matched = {}, 0
    with open(a.log) as f:
        for line in f:
            m = re.search(r"^\[\s*\d+/\s*\d+\]\s+(\S+)\s+-.*?type\s*=\s*(\w+),\s*(?:converting to\s+(\w+)|size)", line)
            if not m:
                continue
            name, src, dst = m.group(1), m.group(2), m.group(3) or m.group(2)
            if name in by_name:
                choice[name] = dst; matched += 1
    weights = category_weights(a.weight)
    picked = []
    for t in tensors:
        typ = choice.get(t.name, t.src_type)
        c = next((c for c in t.cands if c[0].lower() == typ.lower()), None)
        if c is None:
            c = (typ, 0, float("nan"), 0.0, float("nan"))
        picked.append(c)
    s = summarize(tensors, picked, a.objective, weights, p["bytes_other"])
    print(f"llama-quantize's plan from {a.log}: {matched}/{len(tensors)} tensors matched")
    print_summary(s)
    if a.out:
        write_recipe(a.out, tensors, picked, {"source": a.log})


def cmd_apply(a):
    binary = a.quantize_bin or shutil.which("llama-quantize") or "llama-quantize"
    recipe = read_recipe(a.recipe)
    hist = {}
    for typ in recipe.values():
        hist[typ] = hist.get(typ, 0) + 1
    base = a.ftype or max(hist.items(), key=lambda kv: kv[1])[0]
    # llama-quantize's ftype names: Q4_K -> Q4_K_M etc. is not automatic; a pure type name is accepted for most.
    cmd = [binary]
    if a.imatrix:
        cmd += ["--imatrix", a.imatrix]
    cmd += ["--tensor-type-file", a.recipe]
    if a.allow_requantize:
        cmd += ["--allow-requantize"]
    for extra in a.extra or []:
        cmd += extra.split()
    cmd += [a.model, a.out, base]
    if a.threads:
        cmd += [str(a.threads)]
    print("+ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"llama-quantize failed with rc={rc}")
    print(f"wrote {a.out}: {fmt_bytes(os.path.getsize(a.out))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("plan", help="allocate types under a budget")
    s.add_argument("--probe", required=True)
    s.add_argument("--budget", required=True, help="total bytes for the model file, e.g. 6.5G")
    s.add_argument("--reserve", default="0", help="bytes to keep free out of the budget (KV cache, buffers)")
    s.add_argument("--types", help="comma-separated allowed types, e.g. Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_0")
    s.add_argument("--keep", action="append", help="pin a category or name regex to a type: output=Q6_K")
    s.add_argument("--weight", action="append", help="category weight in the objective: ffn_down=2.0")
    s.add_argument("--objective", choices=["sse", "rel", "mse"], default="sse")
    s.add_argument("--solver", choices=["dp", "lagrangian"], default="dp")
    s.add_argument("--unit", default="256K", help="dp granularity")
    s.add_argument("--no-cross-check", dest="cross_check", action="store_false")
    s.add_argument("-o", "--out", default="recipe.txt")
    s.add_argument("--report")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("pareto", help="objective vs budget over a range")
    s.add_argument("--probe", required=True)
    s.add_argument("--from", dest="frm", required=True)
    s.add_argument("--to", required=True)
    s.add_argument("--steps", type=int, default=9)
    s.add_argument("--types")
    s.add_argument("--weight", action="append")
    s.add_argument("--objective", choices=["sse", "rel", "mse"], default="sse")
    s.set_defaults(fn=cmd_pareto)

    s = sub.add_parser("import-llama", help="score llama-quantize's own per-tensor choice from its log")
    s.add_argument("--probe", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--weight", action="append")
    s.add_argument("--objective", choices=["sse", "rel", "mse"], default="sse")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_import_llama)

    s = sub.add_parser("apply", help="run llama-quantize with the recipe")
    s.add_argument("--model", required=True)
    s.add_argument("--recipe", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--imatrix")
    s.add_argument("--ftype", help="base ftype for llama-quantize (default: the recipe's most common type)")
    s.add_argument("--quantize-bin")
    s.add_argument("--threads", type=int)
    s.add_argument("--allow-requantize", action="store_true")
    s.add_argument("--extra", action="append", help="extra llama-quantize arguments, quoted")
    s.set_defaults(fn=cmd_apply)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
