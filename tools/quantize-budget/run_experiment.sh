#!/usr/bin/env bash
# The acceptance experiment for quantize-budget, end to end on one model:
#   1. an imatrix from calibration text
#   2. the probe table
#   3. llama-quantize's own Q4_K_M as the reference, and its size as the budget
#   4. a budget plan at exactly that size, applied
#   5. perplexity (and KL vs the f16 source) for both
# The claim under test: at equal bytes, the planned mixture is at least as good as llama.cpp's hand-tuned mixture.
set -euo pipefail

MODEL=${MODEL:?f16 gguf}
CALIB=${CALIB:?calibration text}
EVAL=${EVAL:-$CALIB}
BIN=${BIN:?llama.cpp build dir with bin/}
OUT=${OUT:-$(dirname "$MODEL")/qb}
REF_FTYPE=${REF_FTYPE:-Q4_K_M}
THREADS=${THREADS:-8}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-40}
TYPES=${TYPES:-Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,Q8_0}
TOOL=$(dirname "$0")/quantize_budget.py
mkdir -p "$OUT"

log() { printf '\n== %s ==\n' "$*"; }

log "1. imatrix"
[ -f "$OUT/imatrix.gguf" ] || "$BIN/bin/llama-imatrix" -m "$MODEL" -f "$CALIB" -o "$OUT/imatrix.gguf" -t "$THREADS" -c "$CTX" --chunks "$CHUNKS" > "$OUT/imatrix.log" 2>&1
echo "imatrix: $(du -h "$OUT/imatrix.gguf" | cut -f1)"

log "2. probe"
[ -f "$OUT/probe.json" ] || "$BIN/bin/llama-quant-probe" -m "$MODEL" --imatrix "$OUT/imatrix.gguf" --types "$TYPES,IQ4_XS,IQ3_S,IQ2_S" --rows 256 -t "$THREADS" -o "$OUT/probe.json" 2> "$OUT/probe.log"
echo "probe: $(python3 -c "import json;p=json.load(open('$OUT/probe.json'));print(len(p['tensors']),'tensors')")"

log "3. reference: llama-quantize $REF_FTYPE"
[ -f "$OUT/ref.gguf" ] || "$BIN/bin/llama-quantize" --imatrix "$OUT/imatrix.gguf" "$MODEL" "$OUT/ref.gguf" "$REF_FTYPE" "$THREADS" > "$OUT/ref.log" 2>&1
REF_BYTES=$(stat -c %s "$OUT/ref.gguf")
echo "reference: $REF_BYTES bytes ($(du -h "$OUT/ref.gguf" | cut -f1))"
python3 "$TOOL" import-llama --probe "$OUT/probe.json" --log "$OUT/ref.log" -o "$OUT/ref-recipe.txt" | tee "$OUT/ref-plan.txt"

log "4. plan at the reference size, apply"
python3 "$TOOL" plan --probe "$OUT/probe.json" --budget "$REF_BYTES" --types "$TYPES" -o "$OUT/recipe.txt" --report "$OUT/plan.json" | tee "$OUT/plan.txt"
[ -f "$OUT/plan.gguf" ] || python3 "$TOOL" apply --model "$MODEL" --recipe "$OUT/recipe.txt" --imatrix "$OUT/imatrix.gguf" --out "$OUT/plan.gguf" --quantize-bin "$BIN/bin/llama-quantize" --threads "$THREADS" > "$OUT/apply.log" 2>&1
echo "plan: $(stat -c %s "$OUT/plan.gguf") bytes ($(du -h "$OUT/plan.gguf" | cut -f1))"

log "5. perplexity + KL vs f16"
[ -f "$OUT/f16.kld" ] || "$BIN/bin/llama-perplexity" -m "$MODEL" -f "$EVAL" -c "$CTX" --chunks "$CHUNKS" -t "$THREADS" --kl-divergence-base "$OUT/f16.kld" > "$OUT/ppl-f16.log" 2>&1
for v in ref plan; do
  "$BIN/bin/llama-perplexity" -m "$OUT/$v.gguf" -f "$EVAL" -c "$CTX" --chunks "$CHUNKS" -t "$THREADS" --kl-divergence-base "$OUT/f16.kld" --kl-divergence > "$OUT/ppl-$v.log" 2>&1 || true
  echo "--- $v ($(stat -c %s "$OUT/$v.gguf") bytes)"
  grep -E 'Mean PPL|Mean KLD|Maximum KLD|Same top p' "$OUT/ppl-$v.log" | head -6
done
grep -E 'Final estimate|estimate' "$OUT/ppl-f16.log" | tail -1 | sed 's/^/f16: /'
