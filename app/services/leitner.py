"""Leitner-box scheduling (PLAN.md "Leitner rule").

Pure functions with the clock passed in, so every box transition is unit-tested
without a database or waiting for real days to pass.
"""

from datetime import datetime, timedelta

MIN_BOX = 1
MAX_BOX = 5
# Days until the next review once a card lands in each box. Doubling spacing is
# the classic Leitner schedule: each correct answer roughly doubles how long the
# word must be remembered before it is asked again.
INTERVAL_DAYS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


def schedule(box: int, correct: bool, now: datetime) -> tuple[int, datetime]:
    """Return (new box, next review time) after one answer."""
    if not MIN_BOX <= box <= MAX_BOX:
        raise ValueError(f"box must be {MIN_BOX}-{MAX_BOX}, got {box}")
    # Wrong goes all the way back to box 1, not down one box: the simple Leitner
    # variant in PLAN.md, and a forgotten word is due again tomorrow.
    new_box = min(box + 1, MAX_BOX) if correct else MIN_BOX
    return new_box, now + timedelta(days=INTERVAL_DAYS[new_box])
