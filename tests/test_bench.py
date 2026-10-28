from model_service.bench import summarize


def test_summarize_excludes_first_call_from_warm_stats():
    summary = summarize([500.0, 10.0, 20.0, 30.0])

    assert summary.first_ms == 500.0
    assert summary.n_warm == 3
    assert summary.median_ms == 20.0
    assert 20.0 < summary.p95_ms <= 30.0


def test_summarize_single_sample_has_no_warm_stats():
    summary = summarize([500.0])

    assert summary.n_warm == 0
    assert summary.median_ms is None
    assert summary.p95_ms is None
