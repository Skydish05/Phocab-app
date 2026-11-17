from datetime import UTC, datetime, timedelta

import pytest

from app.services.leitner import schedule

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("box", "correct", "new_box", "days"),
    [
        (1, True, 2, 2),
        (2, True, 3, 4),
        (3, True, 4, 8),
        (4, True, 5, 16),
        (5, True, 5, 16),  # the top box can't be passed; the card just stays there
        (1, False, 1, 1),
        (3, False, 1, 1),  # wrong resets all the way, not one box down
        (5, False, 1, 1),
    ],
)
def test_schedule(box: int, correct: bool, new_box: int, days: int):
    assert schedule(box, correct, NOW) == (new_box, NOW + timedelta(days=days))


@pytest.mark.parametrize("box", [0, 6])
def test_schedule_rejects_impossible_boxes(box: int):
    with pytest.raises(ValueError):
        schedule(box, True, NOW)
