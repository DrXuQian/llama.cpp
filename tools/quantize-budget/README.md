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

## Roadmap

1. per-tensor allocation under a budget (this) — dense and MoE alike, since an `_exps` tensor is one tensor
2. loss-aware sensitivity: per-layer KL probes to calibrate the category weights, then HAWQ-v3-style curvature
3. per-expert precision inside an `_exps` tensor — needs the runtime to accept mixed-type expert groups
