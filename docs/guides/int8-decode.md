# INT8 Decode

AstrAI's INT8 decode feature is an optional W8A8 inference optimization for
CUDA BF16 models. It is intended for deployments with enough GPU memory to
keep an INT8 cache in addition to the original BF16 model weights.

## Design

The model remains BF16 by default. When enabled, eligible `Linear` modules
prepare a static per-output-channel INT8 weight and use dynamic symmetric INT8
activation quantization only for a single-token decode (`M=1`). All other
paths retain the BF16 implementation:

```text
prefill                  -> BF16 Linear/GEMM
decode, batch > 1        -> BF16 Linear/GEMM
decode, batch = 1        -> INT8 W8A8 GEMV when the module is eligible
```

This preserves the original BF16 fallback and avoids changing training or
checkpoint semantics. INT8 weights are inference-only, non-persistent caches;
they are not written to `state_dict` and increase GPU memory usage.

Eligibility is a capability of the owning model component, not a global shape
list inside `Linear`. The current dense MLP projections and `lm_head` opt in;
attention projections and MoE experts remain opt-out until separately
validated.

If a deployment already has an INT8 weight and scale, call
`Linear.set_int8_weights(weight_q, scale_w)` before inference. The cache is
attached directly and the BF16-to-INT8 quantization step is skipped. The
weight must be `[out_features, in_features]` INT8 and the scale must be FP32
`[out_features]` on the model device.

Enable the path with:

```bash
python scripts/tools/benchmark.py --ckpt params --int8-decode
python scripts/tools/server.py --param_path params --int8-decode
```

The default remains BF16. The experimental `--int8-attention` option is not
recommended: independent activation quantization for each attention projection
adds launch overhead and has not shown an end-to-end benefit.

## Benchmark levels

Use the narrowest benchmark that answers the question:

1. **Kernel level** — `scripts/tools/benchmark_int8.py` compares BF16
   `F.linear`, activation quantization, INT8 GEMM, and fused INT8 Linear on
   explicit `(M,K,N)` shapes.
2. **Model decode** — `scripts/tools/benchmark.py --decode-only` measures
   repeated model decode with scheduler/engine options and can enable or
   disable CUDA Graphs.
3. **End to end** — `scripts/tools/benchmark.py --end-to-end` includes prompt
   prefill, sampling, scheduling, and generation. Its default is BF16; add
   `--int8-decode` for the optional path.
4. **Real testdata** — `scripts/tools/benchmark_testdata.py` measures JSONL
   prompts. It defaults to BF16, accepts `--int8-decode` for one selected mode,
   and accepts `--compare-int8` for a paired accuracy/performance comparison.

Report CUDA Graph, batch size, prompt/output lengths, trial count, and whether
the result is a profiler run. Decode metrics for batch sizes above one should
distinguish batch-step latency from aggregate output-token throughput.

## When to enable

Enable INT8 decode when the workload is dominated by batch-1 generation and
the GPU has sufficient headroom for the additional cached weights and CUDA
Graph allocations. Keep the default BF16 path for memory-constrained devices,
training, prefill-heavy workloads, or batch-oriented serving until those paths
have their own validated INT8 kernels.
