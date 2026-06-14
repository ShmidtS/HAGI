"""Analyze PyTorch profiler trace and print top ops by GPU time."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_trace(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])

    # Collect CUDA kernel times
    cuda_ops = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    cpu_ops = defaultdict(lambda: {"count": 0, "total_us": 0.0})

    total_cuda_us = 0.0
    total_cpu_us = 0.0

    for event in trace_events:
        name = event.get("name", "")
        dur = event.get("dur", 0)
        cat = event.get("cat", "")
        ph = event.get("ph", "")

        if ph != "X":
            continue

        if "cuda" in cat.lower() or "gpu" in cat.lower():
            cuda_ops[name]["count"] += 1
            cuda_ops[name]["total_us"] += dur
            total_cuda_us += dur
        elif cat in {"cpu_op", "Operator", "op", "autograd", "forward", "backward"}:
            cpu_ops[name]["count"] += 1
            cpu_ops[name]["total_us"] += dur
            total_cpu_us += dur

    # Also check for kernel names
    for event in trace_events:
        name = event.get("name", "")
        dur = event.get("dur", 0)
        cat = event.get("cat", "")
        if cat == "Kernel":
            cuda_ops[name]["count"] += 1
            cuda_ops[name]["total_us"] += dur
            total_cuda_us += dur

    print(f"=== Trace: {path.name} ===")
    print(f"Total CUDA/GPU time: {total_cuda_us / 1e6:.2f}s")
    print(f"Total CPU time: {total_cpu_us / 1e6:.2f}s")

    print("\n--- Top 20 CUDA kernels by time ---")
    sorted_cuda = sorted(cuda_ops.items(), key=lambda x: x[1]["total_us"], reverse=True)
    for name, stats in sorted_cuda[:20]:
        pct = 100 * stats["total_us"] / max(total_cuda_us, 1)
        print(
            f"  {stats['total_us']/1000:.1f}ms ({pct:.1f}%) {stats['count']}x | {name[:80]}"
        )

    print("\n--- Top 20 CPU ops by time ---")
    sorted_cpu = sorted(cpu_ops.items(), key=lambda x: x[1]["total_us"], reverse=True)
    for name, stats in sorted_cpu[:20]:
        pct = 100 * stats["total_us"] / max(total_cpu_us, 1)
        print(
            f"  {stats['total_us']/1000:.1f}ms ({pct:.1f}%) {stats['count']}x | {name[:80]}"
        )

    # Memory stats if available
    mem_events = [
        e
        for e in trace_events
        if e.get("ph") == "i" and "memory" in e.get("name", "").lower()
    ]
    if mem_events:
        print(f"\nMemory events: {len(mem_events)}")


if __name__ == "__main__":
    trace_dir = Path(".omc/ultragoal")
    traces = sorted(trace_dir.glob("*.pt.trace.json"))
    if not traces:
        print("No trace files found in .omc/ultragoal/")
        sys.exit(1)
    for trace in traces:
        analyze_trace(trace)
        print("\n" + "=" * 60 + "\n")
