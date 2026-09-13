"""Study package (Phase r1 §2 restructure).

* ``study_config``      — StudyConfig loading/validation (former
                          ``core/study.py``).
* ``date_resolution``   — compact-date parsing/formatting, ET day
                          boundaries, artifact-date extraction (former
                          ``core/dates.py``).
* ``warmup``            — warmup eligibility (former ``core/warmup.py``).
"""

from core.study.study_config import (  # noqa: F401
    STUDY_KINDS,
    StudyConfig,
    StudyConfigError,
    load_study,
)
from core.study.date_resolution import (  # noqa: F401
    DATE_FMT,
    DATE_LEN,
    ET,
    DateError,
    UTC,
    artifact_date_from_filename,
    compact_date_range,
    format_compact_date,
    is_evening_start_et,
    is_valid_compact_date,
    iso_to_eastern_display,
    nearest_valid_date,
    now_et,
    parse_compact_date,
    shift_compact_date,
    today_et,
)
from core.study.warmup import (  # noqa: F401
    WarmupStatus,
    evaluate_warmup,
    teams_ready_for_display,
    warmup_complete_date,
)

__all__ = [
    "STUDY_KINDS", "StudyConfig", "StudyConfigError", "load_study",
    "DATE_FMT", "DATE_LEN", "ET", "UTC", "DateError",
    "artifact_date_from_filename", "compact_date_range",
    "format_compact_date", "is_evening_start_et", "is_valid_compact_date",
    "iso_to_eastern_display", "nearest_valid_date", "now_et",
    "parse_compact_date", "shift_compact_date", "today_et",
    "WarmupStatus", "evaluate_warmup", "teams_ready_for_display",
    "warmup_complete_date",
]
