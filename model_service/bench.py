"""Run the classifier on a folder of photos; print top-k labels, latency, peak memory.

Usage: python -m model_service.bench eval/photos [--top-k 3]

Runs in-process rather than over HTTP so the numbers measure the model alone
(decode + preprocess + inference), and peak RSS is exactly what the model
container needs, which is the number that sizes the EC2 instance.
"""

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

from model_service.classifier import SiglipClassifier, load_labels, prepare_image
from model_service.main import Settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux but bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"No JPEG/PNG/WebP photos in {args.folder}")

    settings = Settings()
    start = time.perf_counter()
    classifier = SiglipClassifier(settings.model_id, load_labels(settings.labels_path))
    load_s = time.perf_counter() - start
    rss_after_load = peak_rss_mb()

    latencies_ms = []
    for photo in photos:
        data = photo.read_bytes()
        start = time.perf_counter()
        predictions = classifier.classify(prepare_image(data), top_k=args.top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)
        labels = ", ".join(f"{p.label_en} ({p.score:.3f})" for p in predictions)
        print(f"{photo.name:<28} {latency_ms:7.1f} ms  {labels}")

    # The first call pays one-off costs (lazy allocations, kernel selection),
    # so it is reported apart from the steady-state numbers a user would see.
    warm = latencies_ms[1:]
    print()
    print(f"model           {settings.model_id}")
    print(f"labels          {len(classifier.labels)}")
    print(f"load time       {load_s:.1f} s")
    print(f"first image     {latencies_ms[0]:.1f} ms")
    if warm:
        print(f"median (warm)   {statistics.median(warm):.1f} ms  (n={len(warm)})")
    if len(warm) >= 2:
        p95 = statistics.quantiles(warm, n=20, method="inclusive")[18]
        print(f"p95 (warm)      {p95:.1f} ms  (n={len(warm)}; near the max at this sample size)")
    print(f"peak RSS load   {rss_after_load:.0f} MB")
    print(f"peak RSS total  {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
