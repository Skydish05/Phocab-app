"""Run the classifier on a folder of photos; print top-k labels, latency, memory.

Usage: python -m model_service.bench eval/photos [--top-k 3] [--threads N]
       [--repeat N] [--output eval/results/bench.md]

Runs in-process rather than over HTTP so the numbers measure the model alone
(decode + preprocess + inference), and memory is exactly what the model
container needs, which is the number that sizes the EC2 instance.

To approximate a small instance, pin CPUs too, e.g. a 2-vCPU EC2 box is one
physical core with 2 hyperthreads:  taskset -c 0,1 python -m ... --threads 2
"""

import argparse
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from model_service.classifier import SiglipClassifier, load_labels, prepare_image
from model_service.main import Settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class LatencySummary:
    first_ms: float
    n_warm: int
    median_ms: float | None
    p95_ms: float | None


def summarize(latencies_ms: list[float]) -> LatencySummary:
    # The first call pays one-off costs (lazy allocations, kernel selection),
    # so it is reported apart from the steady-state numbers a user would see.
    warm = latencies_ms[1:]
    return LatencySummary(
        first_ms=latencies_ms[0],
        n_warm=len(warm),
        median_ms=statistics.median(warm) if warm else None,
        p95_ms=statistics.quantiles(warm, n=20, method="inclusive")[18] if len(warm) >= 2 else None,
    )


def peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux but bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def current_rss_mb() -> float | None:
    # Peak alone hides that SigLIP's startup label encoding is a one-off spike;
    # current RSS is what the service holds while serving. Linux only (/proc).
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def cpu_name() -> str:
    # platform.processor() is just "x86_64" on Linux, useless for comparing runs.
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def usable_cpus() -> int:
    # Records taskset pinning in the report; the command line alone can't show it.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 0


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--threads", type=int, help="torch intra-op threads (default: torch's choice)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="passes over the folder; more samples make p95 meaningful with few photos",
    )
    parser.add_argument("--output", type=Path, help="also write the summary as Markdown here")
    args = parser.parse_args()

    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"No JPEG/PNG/WebP photos in {args.folder}")

    import torch  # deferred like in classifier.py; only needed for set_num_threads

    if args.threads:
        torch.set_num_threads(args.threads)

    settings = Settings()
    start = time.perf_counter()
    classifier = SiglipClassifier(settings.model_id, load_labels(settings.labels_path))
    load_s = time.perf_counter() - start
    peak_after_load = peak_rss_mb()

    latencies_ms = []
    for run in range(args.repeat):
        for photo in photos:
            data = photo.read_bytes()
            start = time.perf_counter()
            predictions = classifier.classify(prepare_image(data), top_k=args.top_k)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            if run == 0:  # labels are identical on every pass; print them once
                labels = ", ".join(f"{p.label_en} ({p.score:.3f})" for p in predictions)
                print(f"{photo.name:<28} {latencies_ms[-1]:7.1f} ms  {labels}")

    summary = summarize(latencies_ms)
    rss_now = current_rss_mb()
    rows = [
        ("model", settings.model_id),
        ("labels", str(len(classifier.labels))),
        ("photos x repeat", f"{len(photos)} x {args.repeat}"),
        ("torch threads", str(torch.get_num_threads())),
        ("CPU", cpu_name()),
        ("usable CPUs", f"{usable_cpus()} of {os.cpu_count()}"),
        ("load time", f"{load_s:.1f} s"),
        ("first image", fmt_ms(summary.first_ms)),
        ("median (warm)", f"{fmt_ms(summary.median_ms)} (n={summary.n_warm})"),
        ("p95 (warm)", f"{fmt_ms(summary.p95_ms)} (n={summary.n_warm})"),
        ("peak RSS after load", f"{peak_after_load:.0f} MB"),
        ("peak RSS total", f"{peak_rss_mb():.0f} MB"),
        ("current RSS at end", "n/a" if rss_now is None else f"{rss_now:.0f} MB"),
    ]
    print()
    for name, value in rows:
        print(f"{name:<20} {value}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = "python -m model_service.bench " + " ".join(sys.argv[1:])
        lines = [
            f"# Model benchmark, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
            "",
            f"Command: `{command}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *(f"| {name} | {value} |" for name, value in rows),
            "",
        ]
        args.output.write_text("\n".join(lines))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
