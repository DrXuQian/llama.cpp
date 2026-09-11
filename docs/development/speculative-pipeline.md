# Speculative GPU execution

`src/llama-spec-pipeline.cpp` owns the internal NextN execution path used by the supported MTP and EAGLE3 models. `llama_context` delegates its existing NextN entry points to this module. The public `llama.h` API and the server decode order are unchanged.

## Ownership

Each context owns one `llama_spec_pipeline`. Draft graphs, verification handoffs, feature snapshots, target graph banks, output copies and completion events stay in that object. Graph bundles retain their scheduler, graph result, tensor metadata, allocations and sampler clones together.

The normal context scheduler temporarily owns an active prefetched target graph. `consume_target()` transfers the prepared graph into the context and commits the CPU recurrent-state mirror. `release_target()` returns it to its bank before an ordinary graph is selected. Neither operation submits another model evaluation.

The context synchronizes its work and waits for cross-context snapshots before destroying the pipeline. The pipeline is destroyed before the backends. Reservation, graph invalidation and output reset have separate hooks because they retire different resources; they are not interchangeable cache-clearing operations.

| Hook | Responsibility |
| --- | --- |
| `reset_targets()` / `reset_draft()` | Retire graph bundles after scheduler synchronization; preserve compatible target graphs on sampler-only reservation. |
| `invalidate_graphs()` | Drain pending work and invalidate target build controls before graph reservation. |
| `begin_decode()` | Retire an unused prefetched target and clear per-batch output state. |
| `stage_outputs()` | Retain deferred output copies and their memory context, or submit ordinary copies immediately. |
| `synchronize()` | Submit deferred copies and wait for output events; full synchronization also drains the copy backend. |
| `wait_for_snapshots()` | Wait for copies produced by the other context before releasing host destinations. |

`common_speculative_nextn_driver` in `common/speculative.cpp` owns the shared sampler installation, pipeline configuration and deferred draft collection. MTP and EAGLE3 keep their model-specific feature handling and catch-up alignment. EAGLE3 uses three target features and a shifted token/position relationship. The MTP implementation also supports shared memory and multiple trained heads; these modes must not inherit assumptions from the single-head Qwen path.

## Execution dependencies

The fast path composes target verification, catch-up, acceptance and the following draft in one CUDA graph. It submits the next target evaluation before the CPU reconciles the previous acceptance result. Draft confidence and target-width selection remain GPU operations.

Alternating graph banks keep outputs alive while a separate backend stream copies snapshots to pinned host buffers. A producer event orders the copies after computation, and a per-bank completion event protects storage reuse. Target layer features use the same output queue. A composed draft consumes its own hidden input; the ordinary handoff copy is deferred until that draft cannot be adopted. Snapshot validation and the compatibility fallbacks remain in place.

There are still CPU batch, position and KV mirror checks between calls. The module is not yet a self-contained, repeatedly replayed GPU cycle, and output collection is not a nonblocking `try_collect` API. Moving code into this module does not remove those dependencies or prove zero GPU idle.

### Device control feedback

`llama_nextn_control` owns a persistent device allocation and a pinned snapshot. Each target bank shares one control buffer across its width variants. The buffer stores draft seed position, accepted count, kept/executed draft counts, previous kept count, epoch, step and seed/proposal tokens.

The draft graph derives its next position and acceptance from the previous device control. The next target receives these outputs in one scalar gather. Its recurrent rollback uses the device previous-kept count, and its conditional width selector uses the kept count. A subsequent draft copies that control directly from the completed target bank. Static acceptance weights and position offsets are uploaded when the graph is built.

Ordinary decode or a broken prefetch chain initializes a new epoch from the host batch. The GPU increments the step within that epoch. CPU reconciliation checks the snapshot's epoch, step, accepted count, position and seed instead of repeating the acceptance scan on the supported target-prefetch path. The fallback without a prepared target retains its previous validation.

Snapshot completion has its own event, and control storage remains alive until pending copies finish. Replacing a bank's control allocation invalidates the graphs that captured its old address. These buffers must not come from temporary graph or CUDA pool allocations.

The CUDA graph cache skips leading input views when choosing its tensor anchor. Otherwise, width variants that share a control view and have the same node count can replace each other's captured graph and force repeated capture during decoding.

This feedback does not yet include token quota, EOS, cancellation, output credits or autonomous multi-round replay. CPU KV metadata, the MTP draft rejection-mask mapping and graph preparation still run between submissions. Snapshot checks are blocking, and the control gather is still submitted separately from the draft graph.

Any future multi-round replay must retain recurrent rollback, EOS/token limits, context capacity and output-buffer lifetime. Neither graph composition nor asynchronous copies alone prove zero GPU idle. Changes to this execution path need measured reductions in the remaining gaps and a separate plain-throughput comparison.

## Extension boundaries

| Change | Required inputs and outputs |
| --- | --- |
| New draft head | Model features, seed alignment, bounded proposal tokens and any required probabilities. |
| Confidence stop or dynamic draft budget | Device budget and executed/kept counts; no CPU length read before target submission. |
| Magic MTP | Correct proposal probabilities, random state, block acceptance and residual sampling. |
| Tree attention | Parent topology, ancestor-only mask and accepted-path state commit. Recurrent models also need per-node state inheritance and rollback. |

These remain private graph/tensor contracts. New algorithms should reuse execution and output ownership without changing the server decode sequence or introducing per-step CPU callbacks. Magic MTP and tree execution for recurrent targets remain subsequent work.

CUDA builds using the CUB argsort fallback select small vocabulary candidate sets in two GPU passes. For one row, 1024 < vocabulary size <= 262144 and k <= 16, each 2048-element tile retains its top k, then one block selects the final sorted candidates. Stable radix sorting preserves the full sort's candidate order, including ties, signed zeros, infinities and NaNs. The existing top-10 draft confidence calculation is unchanged. This backend optimization applies automatically to eligible chain and tree operations and introduces no host readback or synchronization; other sizes and backends retain their existing paths.

Ordinary KV attention can consume the token, position, KV-index and mask tensors in `llm_graph_nextn_target`. This permits ancestor-only masks for a tree verification graph. The caller must keep these tensors alive and schedule their computation before model operations. Materializing positions inside the first RoPE sequence can prevent CUDA fusion; 64-bit KV indices preserve the existing fused RoPE/cache-write path. CUDA conversion from floating positions to 64-bit indices avoids an unsupported direct 32-bit integer conversion.

Host code that reads encoder output must use the public `llama_get_embeddings_nextn()` getter or finish the output copies before reading the internal buffer. The internal context getter does not synchronize pending device-to-host copies. GPU consumers should retain device tensors and use graph dependencies.

## Experimental EAGLE3 tree

`--spec-eagle3-tree` enables fixed-budget tree verification for eligible Qwen3/Qwen3MoE targets with an EAGLE3 head. It requires the existing single-slot, full-GPU, greedy prefetch path, ordinary unquantized fixed KV, `--spec-draft-n-max 3` or `7`, and `--spec-draft-p-min 0`. The target budget includes the root: draft maximum 3 uses K=4, and 7 uses K=8. The default draft maximum remains 3. Unsupported settings retain chain verification. `--no-spec-gpu-pipeline` also disables tree execution. No optimization environment variable is used. The startup command's normal GPU pipeline defaults remain unchanged; tree is experimental because throughput depends on the request.

For K=4, the draft follows three primary tokens, retaining two candidates per step and a third candidate at the root. Probabilities are normalized over the full vocabulary. Original IDs are `[root, main1, side1, main2, side2, main3, side3, root3]`. The four target rows, including the root, can use `[0,1,2,3]`, `[0,1,3,4]`, `[0,1,3,5]`, or `[0,1,2,7]`. The last topology wins when the third root candidate has greater cumulative probability than `main2`; otherwise strict comparisons of IDs 2, 4 and 5 choose among the original templates. Ties preserve the lower original ID. These templates maximize the sum of draft path probabilities in this bounded comb; they do not search descendants of side candidates.

For K=8, the draft follows seven primary tokens and retains seven candidates at each depth. The 50-node pool contains the root, primary chain and side leaves. The GPU ranks cumulative draft probabilities, selects eight nodes and orders them by original ID. Pairwise comparisons break exact ties by the lower original ID, preserving ancestors even when path probabilities underflow. The selected topology, positions, ancestor mask and accepted-path KV commit all remain GPU operations. This maximizes the score sum in the current pool; runtime empirical acceptance learning and descendants of side branches remain subsequent work.

For a fixed target-row budget, the expected accepted length is the sum of node reach probabilities, excluding the root. A node is reached only when its entire ancestor path is accepted. These probabilities cannot increase along a path, so selecting the highest-scoring available frontier node gives an optimal ancestor-closed subset when the scores are exact reach probabilities. Empirical statistics must condition on accepted ancestors; unobserved descendants are not rejection samples. Draft softmax scores are estimates, not measured acceptance probabilities. Probability calibration should be evaluated on held-out selection utility as well as prediction loss. Equal target-row counts also need throughput checks because draft depth, candidate extraction and KV work can differ.

Acceptance compares all candidates with their parents' sampled tokens in one vector operation. Gathering these match bits through the padded ancestor-path table and summing each path replaces the serial node walk. A path is valid only when all K entries match; repeated leaf entries preserve validity for shorter paths.

Probability normalization subtracts the known maximum logit and reduces exponentials in two stages using existing GGML operators. Only the requested candidate probabilities are returned, but the denominator includes the whole vocabulary. This avoids the cooperative large-vocabulary softmax path that reproduced a later SGEMM memory-check failure on the tested CUDA 12.8 / RTX 5090 setup, including a model-free reproducer. Scalar control retains one packed copy; tree inputs use separate device-copy batches of at most 12 tensors. Neither change adds a CPU probability read or synchronization.

This path executes all configured draft steps. A conditional upper-bound skip after step two was tested for K=4, but its branch-entry idle and mixed throughput results did not justify including it here. Existing confidence stopping remains available on the chain path through `--spec-draft-p-min`; combining it with tree verification needs separate validation.

`llama-spec-tree.h` constructs the ancestor mask, selects the accepted path, and gathers the corresponding KV rows before writing a contiguous accepted prefix. It groups only adjacent equal-shaped KV matrices in the same allocation. Sampled rows, logits and EAGLE features are reordered by the same path. Padding tokens stay inside the vocabulary and force rejection at the path's bonus row. Catch-up uses the linear positions and causal mask after KV commit.

The existing alternating target banks own tree inputs and host snapshots. Resolved proposals join the existing output readback. For both budgets, `common_sampler_sample_and_accept_n` reconciles the GPU-resolved path after the next target has been submitted, so the server retains its decode/process/accept order. Comparing remapped target samples with the original chain proposals would produce inconsistent acceptance and break prefetch reuse. The activation log says `experimental EAGLE3 tree active` and reports the target row count only after a target has been queued successfully. Request boundaries, sampler changes and unsupported batches continue to use the existing prefetch guards.

Context shifts notify speculative implementations through `common_speculative_seq_add` after shifting the same KV range. EAGLE3 moves its deferred boundary position and invalidates the previous verification row count. This also fixes chain decoding with explicit context shifting; the hidden boundary vector itself is retained.

After generation has started, eligible single-slot greedy EAGLE3 tree requests retain their fixed verification budget near the output quota. The bounded sampler overload commits at most the remaining output count, including its final target-sampled token. If truncation changes the accepted length, that request has reached its quota; the existing rollback and release paths discard unused work. This avoids building shorter decode graphs at the request tail. Initial short requests, synthetic sampling, hybrid/recurrent targets and other unsupported settings retain the usual draft-length limits. This does not remove cold graph preparation or GPU kernel-dispatch gaps.

## Validation

Compare fixed requests and sampler changes against a frozen build, including draft/accept counts. Check model teardown, fallback paths and context boundaries. Measure plain throughput separately from profiled runs.

GPU continuity requires the union of kernel, memcpy and memset intervals, plus CPU submission timing. Graph begin/end markers only bound gaps outside graphs; they cannot prove that graph interiors contain no idle. Keep cold preparation and request transitions separate, but do not remove a steady-state KV boundary or unexpected recapture from the measurement window. Report profiler overhead and all remaining measured idle explicitly.
