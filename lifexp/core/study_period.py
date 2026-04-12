"""StudyPeriod dataclass and AgeBasis enum for actuarial experience studies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class AgeBasis(Enum):
    """Actuarial age basis conventions."""

    NEAREST_BIRTHDAY = "nearest"
    LAST_BIRTHDAY = "last"
    NEXT_BIRTHDAY = "next"


@dataclass
class StudyPeriod:
    """Defines the observation window for an experience study.

    Attributes
    ----------
    start_date:
        First day of the study window (inclusive).
    end_date:
        Last day of the study window (inclusive).
    """

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be > start_date ({self.start_date})"
            )
