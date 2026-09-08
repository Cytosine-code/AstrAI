import json
import statistics
import time
from pathlib import Path
from typing import Optional, Union

import click
import torch

from astrai.config import BaseModelConfig, ConfigFactory
from astrai.extension import ATTN_BACKEND, AttentionBackendFactory, attn_backend
from astrai.inference.cache import PagePool, TaskCacheManager
from astrai.inference.engine import InferenceEngine
from astrai.inference.runtime.graph import CudaGraphContext
from astrai.inference.workspace import InferenceWorkspace
from astrai.model import AutoModel, AutoRegressiveLM
from astrai.serialization import adapt_config
from astrai.tokenize import AutoTokenizer

_DTYPES = ["bfloat16", "float16", "float32"]
_CACHES = ["contiguous", "paged"]
_BACKENDS = AttentionBackendFactory.list_registered()

# Default 1B GQA preset matching the project checkpoint architecture.
_DEFAULT_CONFIG = {
    "vocab_size": 100000,
    "hidden_size": 1536,
    "num_hidden_layers": 24,
    "intermediate_size": 6912,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
}


class BenchmarkResult:
    def __init__(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        tokens_per_second: float,
        latency_ms: float,
        metadata: Optional[dict] = None,
    ):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens_per_second = tokens_per_second
        self.latency_ms = latency_ms
        self.metadata = metadata or {}


class GenerationBenchmark:
    def __init__(
        self,
        model: AutoModel,
        config: BaseModelConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
        backend: Union[str, ATTN_BACKEND] = ATTN_BACKEND.CUDA,
        cuda_graph: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type
        self.model = model
        self.config = config
        self.backend = backend
        self.cuda_graph = cuda_graph
        self.tokenizer = tokenizer

    def _make_pool(self, batch_size: int, max_seq_len: int) -> PagePool:
        if self.cache_type == "contiguous":
            n_tokens = None
        elif self.cache_type == "paged":
            # Keep the total token capacity equal to contiguous mode while
            # routing allocation through the shared paged pool.
            n_tokens = batch_size * max_seq_len
        else:
            raise ValueError(f"unsupported cache type: {self.cache_type}")

        return PagePool(
            n_layers=self.config.num_hidden_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
            page_size=1,
            n_tokens=n_tokens,
        )

    @staticmethod
    def _make_workspace(pool: PagePool, config: BaseModelConfig) -> InferenceWorkspace:
        return InferenceWorkspace(
            pool.max_batch_size,
            pool.max_seq_len,
            max_q_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=pool.device,
            dtype=pool.dtype,
        )

    @staticmethod
    def _make_task_cache(pool: PagePool) -> TaskCacheManager:
        return TaskCacheManager(pool)

    def _run_prefill(
        self,
        pool: PagePool,
        task_cache: TaskCacheManager,
        batch_size: int,
        prompt_len: int,
        workspace: InferenceWorkspace,
    ) -> list:
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size * prompt_len,), device=self.device
        )
        position_ids = torch.arange(
            prompt_len, dtype=torch.long, device=self.device
        ).repeat(batch_size)

        task_ids = [f"bench_{i}" for i in range(batch_size)]
        for tid in task_ids:
            task_cache.task_alloc(tid, list(range(prompt_len)))

        kv_cache = task_cache.bind(task_ids, workspace, self.device, start_pos=0)
        with torch.inference_mode(), attn_backend(self.backend):
            self.model(
                input_ids,
                kv_cache=kv_cache,
                position_ids=position_ids,
                fwd="prefill",
            )
        torch.cuda.synchronize()
        return task_ids

    def _run_decode_step(
        self,
        pool: PagePool,
        task_cache: TaskCacheManager,
        task_ids: list,
        seq_len: int,
        workspace: InferenceWorkspace,
    ):
        batch_size = len(task_ids)
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size,), device=self.device
        )
        position_ids = torch.tensor(
            [seq_len] * batch_size, dtype=torch.long, device=self.device
        )
        for tid in task_ids:
            task_cache.task_extend(tid, seq_len)
        kv_cache = task_cache.bind(task_ids, workspace, self.device)
        with torch.inference_mode(), attn_backend(self.backend):
            self.model(
                input_ids,
                kv_cache=kv_cache,
                position_ids=position_ids,
                fwd="decode",
            )

    def run_prefill_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        pool = self._make_pool(batch_size, prompt_length)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        task_ids = [f"bench_prefill_{i}" for i in range(batch_size)]
        for tid in task_ids:
            task_cache.task_alloc(tid, list(range(prompt_length)))

        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (batch_size * prompt_length,),
            device=self.device,
        )
        position_ids = torch.arange(
            prompt_length, dtype=torch.long, device=self.device
        ).repeat(batch_size)
        kv_cache = task_cache.bind(task_ids, workspace, self.device, start_pos=0)

        for _ in range(3):
            with torch.inference_mode(), attn_backend(self.backend):
                self.model(
                    input_ids,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                    fwd="prefill",
                )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_trials):
            with torch.inference_mode(), attn_backend(self.backend):
                self.model(
                    input_ids,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                    fwd="prefill",
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * prompt_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="prefill",
            batch_size=batch_size,
            seq_len=prompt_length,
            tokens_per_second=tps,
            latency_ms=elapsed / num_trials * 1000,
            metadata={"benchmark_type": "prefill", "num_trials": num_trials},
        )

    def run_decoding_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        if self.tokenizer is None:
            raise ValueError("Engine decode benchmark requires a tokenizer")

        # Use the real engine so scheduler, executor, sampling, and graph
        # warmup/replay are included in the measured generation path.
        phrase = "Benchmark the language model with a realistic generation prompt. "
        prompt_ids = self.tokenizer.encode(
            (phrase * (prompt_length // 10 + 2)).strip()
        )[:prompt_length]
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        max_seq_len = prompt_tokens + gen_length
        pool = self._make_pool(batch_size, max_seq_len)
        engine = InferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            cache=pool,
            enable_cuda_graph=self.cuda_graph,
            backend=self.backend,
        )
        prompts = [prompt] * batch_size

        try:
            # Capture graphs and populate the allocator before timing. The
            # first request also includes model/scheduler startup effects.
            engine.generate(prompts, max_tokens=gen_length, temperature=0.0)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(num_trials):
                engine.generate(prompts, max_tokens=gen_length, temperature=0.0)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        finally:
            engine.shutdown()

        tokens = batch_size * gen_length * num_trials
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tokens / elapsed,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "engine_decode",
                "num_trials": num_trials,
                "prompt_length": prompt_tokens,
                "backend": engine.backend_name,
                "cuda_graph": engine.cuda_graph_enabled,
            },
        )

    def run_end_to_end_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        """Measure complete prompt-to-generation and steady-state throughput."""
        if self.tokenizer is None:
            raise ValueError("End-to-end benchmark requires a tokenizer")
        phrase = "Benchmark the language model with a realistic generation prompt. "
        prompt_ids = self.tokenizer.encode(
            (phrase * (prompt_length // 10 + 2)).strip()
        )[:prompt_length]
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        pool = self._make_pool(batch_size, prompt_tokens + gen_length)
        engine = InferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=batch_size,
            max_seq_len=prompt_tokens + gen_length,
            cache=pool,
            enable_cuda_graph=self.cuda_graph,
            backend=self.backend,
        )
        prompts = [prompt] * batch_size
        try:
            list(engine.generate(prompts, stream=True, max_tokens=gen_length, temperature=0.0))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            totals, ttfts, decode_totals, counts = [], [], [], []
            for _ in range(num_trials):
                start = time.perf_counter()
                first = None
                count = 0
                for _item in engine.generate(
                    prompts, stream=True, max_tokens=gen_length, temperature=0.0
                ):
                    count += 1
                    if first is None:
                        first = time.perf_counter()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                total = end - start
                ttft = first - start if first is not None else total
                ttfts.append(ttft)
                decode_totals.append(total - ttft)
                totals.append(total)
                counts.append(count)
        finally:
            engine.shutdown()

        total = statistics.median(totals)
        ttft = statistics.median(ttfts)
        generated = statistics.median(counts)
        decode_tokens = max(generated - batch_size, 1)
        output_tps = decode_tokens / statistics.median(decode_totals)
        return BenchmarkResult(
            name="end_to_end",
            batch_size=batch_size,
            seq_len=prompt_tokens + gen_length,
            tokens_per_second=(prompt_tokens * batch_size + generated) / total,
            latency_ms=total * 1000.0,
            metadata={
                "benchmark_type": "end_to_end",
                "num_trials": num_trials,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "ttft_ms": ttft * 1000.0,
                "output_tps": output_tps,
                "cuda_graph": engine.cuda_graph_enabled,
            },
        )

    def _run_graph_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        task_ids = self._run_prefill(
            pool, task_cache, batch_size, prompt_length, workspace
        )

        b = batch_size
        input_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)
        position_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)

        gctx = CudaGraphContext(enabled=True)
        graph_key = (b,)

        def _decode_graph_step(seq_len):
            input_ids_buf.copy_(
                torch.randint(0, self.config.vocab_size, (b,), device=self.device)
            )
            position_ids_buf[:] = seq_len
            for tid in task_ids:
                task_cache.task_extend(tid, seq_len)
            kv_cache = task_cache.bind(task_ids, workspace, self.device)

            with torch.inference_mode(), attn_backend(self.backend):
                return gctx.forward(
                    self.model,
                    key=graph_key,
                    input_ids=input_ids_buf,
                    kv_cache=kv_cache,
                    position_ids=position_ids_buf,
                    fwd="decode",
                )

        for i in range(5):
            _decode_graph_step(prompt_length + i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            _decode_graph_step(prompt_length + 5 + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
                "cuda_graph": True,
            },
        )

    def _run_plain_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        task_ids = self._run_prefill(
            pool, task_cache, batch_size, prompt_length, workspace
        )

        for i in range(5):
            self._run_decode_step(
                pool, task_cache, task_ids, prompt_length + i, workspace
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            self._run_decode_step(
                pool, task_cache, task_ids, prompt_length + 5 + i, workspace
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
            },
        )


def print_benchmark_result(result: BenchmarkResult) -> None:
    print("-" * 80)
    print(f"{result.name.upper()} — Batch={result.batch_size}, SeqLen={result.seq_len}")
    if result.name == "end_to_end":
        print(f"  Total latency: {result.latency_ms:.2f} ms")
        print(f"  TTFT         : {result.metadata['ttft_ms']:.2f} ms")
        print(f"  Decode TPS  : {result.metadata['output_tps']:.1f} tokens/s")
        print(f"  Total TPS   : {result.tokens_per_second:.1f} tokens/s")
        print(f"  Prompt tokens: {result.metadata['prompt_tokens']}")
        print(f"  Output tokens: {result.metadata['generated_tokens']}")
    else:
        print(f"  Throughput : {result.tokens_per_second:.1f} tokens/s")
        print(f"  Latency    : {result.latency_ms:.2f} ms/step")
    displayed = {
        "benchmark_type",
        "prompt_tokens",
        "generated_tokens",
        "ttft_ms",
        "output_tps",
    }
    for k, v in result.metadata.items():
        if k not in displayed:
            print(f"  {k.replace('_', ' ').title()}: {v}")
    print("-" * 80)


@click.command(name="benchmark", help="Benchmark model throughput and latency.")
@click.option("--device", default="cuda", help="Device.")
@click.option(
    "--dtype", type=click.Choice(_DTYPES), default="bfloat16", help="Data type."
)
@click.option(
    "--cache", type=click.Choice(_CACHES), default="contiguous", help="KV cache type."
)
@click.option(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="cuda",
    help="Attention backend.",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Run both backends and print side-by-side speed comparison.",
)
@click.option("--batch_size", type=int, default=4, help="Batch size.")
@click.option("--prompt_length", type=int, default=512, help="Prompt length.")
@click.option("--gen_length", type=int, default=128, help="Generation length.")
@click.option("--num_trials", type=int, default=5, help="Number of trials.")
@click.option("--prefill_only", is_flag=True, help="Prefill benchmark only.")
@click.option("--decode_only", is_flag=True, help="Decode benchmark only.")
@click.option(
    "--end_to_end",
    "--end-to-end",
    "end_to_end",
    is_flag=True,
    help="Complete prompt-to-generation benchmark only.",
)
@click.option(
    "--int8-decode/--no-int8-decode",
    default=False,
    help="Use cached W8A8 only for supported batch-1 decode MLP projections.",
)
@click.option(
    "--int8-attention/--no-int8-attention",
    default=False,
    help="Also use INT8 for decode q_proj/o_proj attention GEMVs.",
)
@click.option(
    "--cuda-graph/--no-cuda-graph",
    default=True,
    help="Enable or disable CUDA graph capture for engine decode.",
)
@click.option(
    "--ckpt",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Checkpoint directory. If omitted, a randomly-initialized model is "
    "built from --config or the default 1B GQA preset.",
)
@click.option(
    "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional model config JSON (used when --ckpt is omitted to define the "
    "architecture). Defaults to the 1B GQA preset.",
)
def benchmark_command(
    device: str,
    dtype: str,
    cache: str,
    backend: str,
    compare: bool,
    batch_size: int,
    prompt_length: int,
    gen_length: int,
    num_trials: int,
    prefill_only: bool,
    decode_only: bool,
    end_to_end: bool,
    int8_decode: bool,
    int8_attention: bool,
    cuda_graph: bool,
    ckpt: Optional[str],
    config_path: Optional[Path],
) -> None:
    """Benchmark model throughput and latency."""
    dtype_map: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if ckpt is not None:
        click.echo(f"Loading model from {ckpt} ...")
        config = ConfigFactory.load(
            adapt_config(
                json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8-sig"))
            )
        )
        model = AutoModel.from_pretrained(ckpt)
    else:
        raw = dict(_DEFAULT_CONFIG)
        if config_path is not None:
            raw.update(json.loads(config_path.read_text(encoding="utf-8-sig")))
        config = ConfigFactory.load(raw)
        model = AutoRegressiveLM(config)
        click.echo(
            f"Using randomly-initialized model "
            f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)"
        )

    tokenizer = AutoTokenizer.from_pretrained(ckpt or Path("params"))

    model.to(device=device, dtype=dtype_map[dtype])
    model.eval()
    if int8_decode:
        prepared = model.set_int8_decode_enabled(
            True, include_attention=int8_attention
        )
        click.echo(f"INT8 decode: prepared {prepared} supported Linear projections")

    backends = _BACKENDS if compare else [backend]

    for name in backends:
        bench = GenerationBenchmark(
            model=model,
            config=config,
            device=device,
            dtype=dtype_map[dtype],
            cache_type=cache,
            backend=name,
            cuda_graph=cuda_graph,
            tokenizer=tokenizer,
        )

        click.secho(
            f"Benchmark: device={device} dtype={dtype} backend={name}", bold=True
        )

        if end_to_end:
            result = bench.run_end_to_end_benchmark(
                batch_size=batch_size,
                prompt_length=prompt_length,
                gen_length=gen_length,
                num_trials=num_trials,
            )
            print_benchmark_result(result)
            continue

        if not decode_only:
            result = bench.run_prefill_benchmark(
                batch_size=batch_size,
                prompt_length=prompt_length,
                num_trials=num_trials,
            )
            print_benchmark_result(result)

        if not prefill_only:
            result = bench.run_decoding_benchmark(
                batch_size=batch_size,
                prompt_length=prompt_length,
                gen_length=gen_length,
                num_trials=num_trials,
            )
            print_benchmark_result(result)


if __name__ == "__main__":
    benchmark_command()
