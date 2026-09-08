"""Benchmark W8A8 primitives against BF16 linear on model-relevant shapes."""
import argparse
import statistics

import torch
import torch.nn.functional as F

from astrai.extension.loader import is_available
from astrai.extension.ops.int8 import (
    linear_forward_int8,
    mm_int8,
    quantize_dynamic_bf16,
    quantize_weight_bf16,
)


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--shape", action="append", metavar="M,K,N",
        help="benchmark only this shape; may be repeated (for example 16,1536,6912)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or not is_available("int8_ops"):
        raise SystemExit("a built int8_ops CUDA extension is required")
    default_shapes = ((1, 1536, 6912), (16, 1536, 6912),
                      (128, 1536, 6912), (1, 6912, 1536))
    try:
        shapes = tuple(tuple(map(int, item.split(","))) for item in args.shape) \
            if args.shape else default_shapes
    except ValueError as exc:
        parser.error(f"--shape must be M,K,N: {exc}")
    if any(len(shape) != 3 or min(shape) <= 0 for shape in shapes):
        parser.error("--shape must contain three positive integers: M,K,N")

    print("M K N | bf16_us quant_us int8_mm_us int8_linear_us")
    for m, k, n in shapes:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        wq, sw = quantize_weight_bf16(w)
        xq, sx = quantize_dynamic_bf16(x)
        bf16 = timed(lambda: F.linear(x, w), args.warmup, args.iters)
        quant = timed(lambda: quantize_dynamic_bf16(x), args.warmup, args.iters)
        gemm = timed(lambda: mm_int8(xq, wq, sx, sw), args.warmup, args.iters)
        linear = timed(lambda: linear_forward_int8(x, wq, sw), args.warmup, args.iters)
        print(f"{m:3d} {k:4d} {n:4d} | {bf16:7.2f} {quant:8.2f} {gemm:10.2f} {linear:13.2f}")


if __name__ == "__main__":
    main()
