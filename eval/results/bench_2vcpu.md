# Model benchmark, 2026-10-28 01:31 UTC

Command: `python -m model_service.bench eval/photos --threads 2 --repeat 5 --output eval/results/bench_2vcpu.md`

| Metric | Value |
| --- | --- |
| model | google/siglip2-base-patch16-224 |
| labels | 324 |
| photos x repeat | 10 x 5 |
| torch threads | 2 |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics |
| usable CPUs | 2 of 16 |
| load time | 38.4 s |
| first image | 325.9 ms |
| median (warm) | 330.5 ms (n=49) |
| p95 (warm) | 342.0 ms (n=49) |
| peak RSS after load | 2102 MB |
| peak RSS total | 2102 MB |
| current RSS at end | 1594 MB |
