# Model benchmark, 2026-10-28 01:32 UTC

Command: `python -m model_service.bench eval/photos --repeat 5 --output eval/results/bench_laptop_16cpu.md`

| Metric | Value |
| --- | --- |
| model | google/siglip2-base-patch16-224 |
| labels | 324 |
| photos x repeat | 10 x 5 |
| torch threads | 8 |
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics |
| usable CPUs | 16 of 16 |
| load time | 12.8 s |
| first image | 69.6 ms |
| median (warm) | 65.2 ms (n=49) |
| p95 (warm) | 73.8 ms (n=49) |
| peak RSS after load | 2108 MB |
| peak RSS total | 2108 MB |
| current RSS at end | 1602 MB |
