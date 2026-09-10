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
| `synchronize()` | Submit deferred copies and wait for the selected output event or scheduler. |
| `wait_for_snapshots()` | Wait for copies produced by the other context before releasing host destinations. |

`common_speculative_nextn_driver` in `common/speculative.cpp` owns the shared sampler installation, pipeline configuration and deferred draft collection. MTP and EAGLE3 keep their model-specific feature handling and catch-up alignment. EAGLE3 uses three target features and a shifted token/position relationship. The MTP implementation also supports shared memory and multiple trained heads; these modes must not inherit assumptions from the single-head Qwen path.

## Execution dependencies

The existing fast path submits the next target evaluation before the CPU reconciles the previous acceptance result. Draft confidence and target-width selection remain GPU operations. This refactor preserves that behavior and its compatibility fallbacks.

There are still CPU batch, position and KV mirror checks between calls. The module is not yet a self-contained, repeatedly replayed GPU cycle, and output collection is not a nonblocking `try_collect` API. Moving code into this module does not remove those dependencies or prove zero GPU idle.

### Device control feedback

`llama_nextn_control` owns a persistent device allocation and a pinned snapshot. Each target bank shares one control buffer across its width variants. The buffer stores draft seed position, accepted count, kept/executed draft counts, previous kept count, epoch, step and seed/proposal tokens.

The draft graph derives its next position and acceptance from the previous device control. The next target receives these outputs in one scalar gather. Its recurrent rollback uses the device previous-kept count, and its conditional width selector uses the kept count. A subsequent draft copies that control directly from the completed target bank. Static acceptance weights and position offsets are uploaded when the graph is built.

Ordinary decode or a broken prefetch chain initializes a new epoch from the host batch. The GPU increments the step within that epoch. CPU reconciliation checks the snapshot's epoch, step, accepted count, position and seed instead of repeating the acceptance scan on the supported target-prefetch path. The fallback without a prepared target retains its previous validation.

Snapshot completion has its own event, and control storage remains alive until pending copies finish. Replacing a bank's control allocation invalidates the graphs that captured its old address. These buffers must not come from temporary graph or CUDA pool allocations.

The CUDA graph cache skips leading input views when choosing its tensor anchor. Otherwise, width variants that share a control view and have the same node count can replace each other's captured graph and force repeated capture during decoding.

This feedback does not yet include token quota, EOS, cancellation, output credits or autonomous catch-up scheduling. CPU KV metadata, the MTP draft rejection-mask mapping and graph preparation still run between submissions. Snapshot checks are blocking, and the control gather is still submitted separately from the draft graph. Composing those operations and defining bounded output collection remain necessary before a complete GPU cycle can be replayed.

The next execution change must give submit and collect independent responsibilities: submit consumes device seed/length/position state, while collect only publishes a completed output snapshot. A composed cycle must also retain recurrent rollback, EOS/token limits, context capacity and output-buffer lifetime. Adding asynchronous copies alone does not establish that contract.

## Extension boundaries

| Change | Required inputs and outputs |
| --- | --- |
| New draft head | Model features, seed alignment, bounded proposal tokens and any required probabilities. |
| Confidence stop or dynamic draft budget | Device budget and executed/kept counts; no CPU length read before target submission. |
| Magic MTP | Correct proposal probabilities, random state, block acceptance and residual sampling. |
| Tree attention | Parent topology, ancestor-only mask and accepted-path state commit. Recurrent models also need per-node state inheritance and rollback. |

These remain private graph/tensor contracts. New algorithms should reuse execution and output ownership without changing the server decode sequence or introducing per-step CPU callbacks. Magic MTP and tree execution are subsequent work, not features added by this refactor.

## Validation

Compare fixed requests and sampler changes against a frozen build, including draft/accept counts. Check model teardown, fallback paths and context boundaries. Measure plain throughput separately from profiled runs.

GPU continuity requires the union of kernel, memcpy and memset intervals, plus CPU submission timing. Graph begin/end markers only bound gaps outside graphs; they cannot prove that graph interiors contain no idle. Keep cold preparation and request transitions separate, but do not remove a steady-state KV boundary or unexpected recapture from the measurement window. Report profiler overhead and all remaining measured idle explicitly.
