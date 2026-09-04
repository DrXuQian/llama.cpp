# quantize-budget

Pick a quantization type per tensor so the model fits a byte budget with the least estimated damage, then apply
it with `llama-quantize`. Three parts:

| part | what it does | output |
|---|---|---|
| `llama-quant-probe` (C++) | for every weight tensor × every candidate type: bytes, and the imatrix-weighted error, measured with ggml's own quantizers on a uniform sample of rows | `probe.json` |
| `quantize_budget.py plan` | multiple-choice knapsack under the budget (exact DP over 256 KiB units, Lagrangian cross-check) | `recipe.txt` |
| `quantize_budget.py apply` | `llama-quantize --tensor-type-file recipe.txt` | the GGUF |

```
llama-quant-probe -m model-f16.gguf --imatrix im.gguf --types Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_0 -o probe.json
quantize_budget.py plan  --probe probe.json --budget 6.5G --reserve 1.5G --types Q3_K,Q4_K,Q5_K,Q6_K -o recipe.txt
quantize_budget.py apply --model model-f16.gguf --recipe recipe.txt --imatrix im.gguf --out model.gguf
```

`--budget` is the whole file; `--reserve` is what to leave free of it (KV cache, compute buffers). Tensors the
probe does not plan (norms, biases, the router) are counted as they are.

## What the error means, and what it does not

With an importance matrix `m_j = E[x_j²]`, the probe's `sse` for a tensor at a type is
`Σ_i Σ_j m_j (w_ij − q_ij)²` — the expected squared perturbation of the tensor's *output*, `E‖(W−Q)x‖²`, under
a diagonal-Hessian model of the input. It is the objective ggml's imatrix quantizers minimise inside one tensor;
the planner extends it across tensors. Without an imatrix it degrades to plain MSE (what `llama-quantize` does
then too).

It does **not** know how much a perturbation in one tensor costs in final loss — that a `ffn_down` in layer 3 and
an `attn_v` in layer 40 with the same `sse` are not equally harmful. `--weight category=w` is the hook for that,
and llama.cpp's own recipes are the prior (`import-llama` scores them on the same table so the two can be compared
at equal size). Measuring it — HAWQ-style curvature, or per-layer KL probes — is the next layer, not this one.

## Reference comparison

`run_experiment.sh` runs the acceptance check on one model: build llama.cpp's `Q4_K_M`, plan at exactly its size,
and compare perplexity and KL-to-f16 for both. The claim is only "at equal bytes, not worse than the hand-tuned
mixture"; a planner that cannot clear that bar has no business allocating anything finer.

## Measurement (Qwen2.5-0.5B-Instruct, 40×512 tokens; imatrix + sensitivity fitted on wikitext-2 *test*, evaluated on a disjoint wikitext-2 *train* slice)

All four files have exactly `Q4_K_M`'s weight bytes (463 MiB). On this model that mixture is mostly `q5_0`, because
hidden=896 is not a multiple of 256 and every k-quant falls back.

| plan | mean KL vs f16 (held-out) | ΔPPL (held-out) | max KL | mean KL (in-sample) |
|---|---:|---:|---:|---:|
| llama.cpp `Q4_K_M` | 0.0280 ± 0.0004 | +0.679 | 1.37 | 0.0271 |
| **budget plan, measured category weights** | **0.0192 ± 0.0005** | **+0.574** | 2.26 | 0.0195 |
| budget plan, relative-error objective | 0.0432 | +0.926 | 0.64 | 0.0413 |
| budget plan, raw SSE objective | 0.0742 | +1.102 | 4.63 | 0.0710 |

Raw SSE is 2.6× better by its own metric and 2.6× worse by KL: it bought `q8_0` for every `attn_q`/`attn_k`
(cheap bytes) with bits taken from `ffn_down` and the LM head. The measured exchange rates say why — KL per unit of
probe error, median category = 1:

```
attn_output 10.0   ffn_down 9.6   attn_v 6.4   attn_k 3.2   output 1.0   attn_q 0.28   ffn_gate 0.25   ffn_up 0.16   token_embd 0.075
```

A unit of output perturbation before a softmax (`attn_q`, `attn_k`) or before an activation (`ffn_gate`, `ffn_up`)
costs a fraction of one written straight into the residual stream (`attn_output`, `ffn_down`, `attn_v`). With those
weights the planner reproduces llama.cpp's structural choices on its own (`q8_0` for `attn_v` and `output`, `q5_K`/
`q6_K` for `ffn_down`) and does better where the heuristic is flat: 31% less KL at the same size, held out. The
tail (max KL) is somewhat worse; a minimax term is the obvious next knob.

## Roadmap

1. per-tensor allocation under a budget (this) — dense and MoE alike, since an `_exps` tensor is one tensor
2. loss-aware sensitivity: per-category KL probes calibrate the weights (done, `sensitivity.py`); next per-layer
   groups, then HAWQ-v3-style curvature, and a tail (max-KL) term in the objective
3. per-expert precision inside an `_exps` tensor — needs the runtime to accept mixed-type expert groups
