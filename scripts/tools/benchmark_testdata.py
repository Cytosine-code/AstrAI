"""General generation benchmark over JSONL testdata.

By default this measures the model's normal BF16 path. Use ``--int8-decode``
to select the optional INT8 decode path, or ``--compare-int8`` to run paired
BF16/INT8 requests on identical prompts and report output agreement.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


_DEFAULT_PROMPT = (
    "Benchmark the language model with a realistic generation prompt. " * 16
).strip()


@dataclass
class RequestResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    target: str | None = None


def _prompt(row: dict, tokenizer=None) -> str:
    if "prompt" in row:
        value = row["prompt"]
        if isinstance(value, tuple):
            value = "".join(value)
        raw = str(value)
        if tokenizer is not None and getattr(tokenizer, "_chat_template", None) is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return raw
    if "messages" in row:
        if tokenizer is not None and getattr(tokenizer, "_chat_template", None) is not None:
            return tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
        return str(row["messages"])
    context = str(row.get("context", ""))
    question = str(row.get("input", row.get("question", "")))
    return (
        "Answer the question based on the given passages. Only give the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"no JSONL rows found in {path}")
    return rows


def _run_requests(engine, prompts: list[str], max_tokens: int) -> list[RequestResult]:
    results = []
    for prompt in prompts:
        start = time.perf_counter()
        first = None
        pieces = []
        for token in engine.generate(
            prompt, stream=True, max_tokens=max_tokens, temperature=0.0
        ):
            if first is None:
                first = time.perf_counter()
            pieces.append(token)
        end = time.perf_counter()
        total = (end - start) * 1000.0
        text = "".join(pieces)
        # Tokenizer-independent fallback: output text is still comparable;
        # callers replace this count with tokenizer IDs below when available.
        results.append(
            RequestResult(
                text=text,
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=((first or end) - start) * 1000.0,
                total_ms=total,
            )
        )
    return results


def _summarize(label: str, results: list[RequestResult], tokenizer, prompts: list[str]) -> dict:
    for result in results:
        result.output_tokens = max(1, len(tokenizer.encode(result.text, add_special_tokens=False)))
    for result, prompt in zip(results, prompts):
        result.prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    ttft = statistics.median(r.ttft_ms for r in results)
    decode = statistics.median(
        (r.total_ms - r.ttft_ms) / max(r.output_tokens - 1, 1) for r in results
    )
    total = sum(r.total_ms for r in results)
    tokens = sum(r.output_tokens for r in results)
    input_tokens = sum(r.prompt_tokens for r in results)
    elapsed_s = total / 1000.0
    return {
        "label": label,
        "requests": len(results),
        "ttft_ms": ttft,
        "decode_ms": decode,
        "input_tps": input_tokens / elapsed_s,
        "output_tps": tokens / elapsed_s,
        "total_token_tps": (input_tokens + tokens) / elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": tokens,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("params"))
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Optional JSONL input. If omitted, use the short prompt from benchmark.py.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Rows to test; 0 means all.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1, help="Measured repetitions per mode.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--int8-decode", action="store_true",
        help="Enable cached INT8 weights for supported M=1 decode projections.",
    )
    mode.add_argument(
        "--compare-int8", action="store_true",
        help="Run both BF16 and INT8 modes and compare generated outputs.",
    )
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens <= 0 or args.warmup < 0 or args.trials <= 0:
        parser.error("limit must be >= 0, max-tokens/trials > 0, warmup >= 0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    if args.input is None:
        rows = [{"prompt": _DEFAULT_PROMPT}]
    else:
        rows = _load_rows(args.input, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    prompts = [_prompt(row, tokenizer) for row in rows]
    targets = [row.get("target_text") for row in rows]
    model = AutoModel.from_pretrained(args.ckpt).to("cuda", dtype=torch.bfloat16).eval()

    def run_mode(enabled: bool):
        model.set_int8_decode_enabled(enabled)
        # Weight preparation and graph capture must belong to the selected
        # mode; switching a live engine would replay the previously captured
        # BF16 graph and also charge quantization to INT8 TTFT.
        torch.cuda.synchronize()
        max_seq_len = max(
            len(tokenizer.encode(prompt, add_special_tokens=False)) + args.max_tokens
            for prompt in prompts
        )
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=1,
            max_seq_len=max_seq_len,
        )
        try:
            for _ in range(args.warmup):
                _run_requests(engine, prompts[:1], min(args.max_tokens, 8))
            torch.cuda.synchronize()
            measured = []
            for _ in range(args.trials):
                measured.extend(_run_requests(engine, prompts, args.max_tokens))
            return _summarize(
                "INT8" if enabled else "BF16",
                measured,
                tokenizer,
                prompts * args.trials,
            )
        finally:
            engine.shutdown()

    summaries = [run_mode(args.int8_decode)] if not args.compare_int8 else [run_mode(False), run_mode(True)]
    print(f"Testdata benchmark: {args.input or 'benchmark.py short prompt'}")
    print(f"Requests={len(rows)} trials={args.trials} max_tokens={args.max_tokens}")
    for summary in summaries:
        print(
            f"{summary['label']:4s} | TTFT {summary['ttft_ms']:.2f} ms | "
            f"Decode {summary['decode_ms']:.2f} ms/token | "
            f"Input TPS {summary['input_tps']:.2f} | "
            f"Output TPS {summary['output_tps']:.2f} | "
            f"Total-token TPS {summary['total_token_tps']:.2f} | "
            f"Input tokens {summary['input_tokens']} | Output tokens {summary['output_tokens']}"
        )
    if args.compare_int8:
        bf16, int8 = summaries
        if bf16["output_tokens"] == int8["output_tokens"]:
            print(f"INT8 speedup (decode): {bf16['decode_ms'] / int8['decode_ms']:.3f}x")
        else:
            print("INT8 speedup (decode): not comparable (different token counts)")
        agreement = sum(
            _norm(a.text) == _norm(b.text)
            for a, b in zip(bf16["results"], int8["results"])
        ) / max(len(bf16["results"]), 1)
        print(
            "Normalized output agreement: "
            f"{agreement * 100:.2f}% "
            "(case/whitespace-normalized final text)"
        )
    if args.int8_decode:
        print("Mode: INT8 decode enabled (supported M=1 projections only)")
    else:
        print("Mode: BF16 baseline")
    if args.compare_int8:
        bf16, int8 = summaries
    scored = [(idx, target) for idx, target in enumerate(targets) if target is not None]
    if scored:
        if args.compare_int8:
            bf16_acc = sum(_norm(bf16["results"][idx].text) == _norm(str(target)) for idx, target in scored) / len(scored)
            int8_acc = sum(_norm(int8["results"][idx].text) == _norm(str(target)) for idx, target in scored) / len(scored)
            print(f"Target exact-match: BF16 {bf16_acc * 100:.2f}% | INT8 {int8_acc * 100:.2f}%")
        else:
            label = summaries[0]["label"]
            accuracy = sum(_norm(summaries[0]["results"][idx].text) == _norm(str(target)) for idx, target in scored) / len(scored)
            print(f"Target exact-match: {label} {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
