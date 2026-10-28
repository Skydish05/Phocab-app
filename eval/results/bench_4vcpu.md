# Model benchmark, 2026-10-28 01:32 UTC

Command: `python -m model_service.bench eval/photos --threads 4 --repeat 5 --output eval/results/bench_4vcpu.md`

| Metric | Value |
| --- | --- |
| model | google/siglip2-base-patch16-224 |
| labels | 324 |
| photos x repeat | 10 x 5 |
| torch threads | 4 |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics |
| usable CPUs | 4 of 16 |
| load time | 23.5 s |
| first image | 210.1 ms |
| median (warm) | 206.7 ms (n=49) |
| p95 (warm) | 214.6 ms (n=49) |
| peak RSS after load | 2101 MB |
| peak RSS total | 2101 MB |
| current RSS at end | 1594 MB |
