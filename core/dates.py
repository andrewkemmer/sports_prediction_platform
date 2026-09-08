"""Date parsing, validation, and formatting utilities.

All artifact dates are compact ``YYYYMMDD`` strings (the reference repo's
contract). This module is the single place that parses, validates, and
compares them; it never imports pandas or streamlit so backend code and
tests can use it anywhere.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

DATE_FMT = "%Y%m%d"
DATE_LEN = 8


class DateError(ValueError):
    """Raised when a date string cannot be parsed or is out of contract."""


def parse_compact_date(value: str) -> date:
    """Parse a ``YYYYMMDD`` string into a ``datetime.date``.

    Raises ``DateError`` for None/empty/wrong-length/non-numeric/impossible
    dates. Accepts an ``int`` input (artifacts sometimes carry numeric ids).
    """
    if value is None or isinstance(value, bool):
        raise DateError(f"invalid compact date: {value!r}")
    if isinstance(value, (int, float)):
        s = str(int(value))
    else:
        s = str(value).strip()
    if len(s) != DATE_LEN or not s.isdigit():
        raise DateError(f"invalid compact date: {value!r}")
    try:
        return datetime.strptime(s, DATE_FMT).date()
    except ValueError as exc:
        raise DateError(f"invalid compact date: {value!r}") from exc


def format_compact_date(d: date | datetime) -> str:
    """Format a date (or datetime) as ``YYYYMMDD``."""
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime(DATE_FMT)


def today_et() -> date:
    """Today's date in US Eastern (the platform's slate timezone)."""
    return datetime.now(tz=ET).date()


def now_et() -> datetime:
    """Now in US Eastern."""
    return datetime.now(tz=ET)


def is_valid_compact_date(value: str) -> bool:
    """True when the value parses as a ``YYYYMMDD`` date."""
    try:
        parse_compact_date(value)
        return True
    except DateError:
        return False


def compact_date_range(start: date | str, end: date | str,
                       *, inclusive: bool = True) -> list[str]:
    """List compact dates from start to end (newest last).

    Raises ``DateError`` when start > end. With ``inclusive=False`` the end
    date is excluded.
    """
    d0 = start if isinstance(start, date) else parse_compact_date(start)
    d1 = end if isinstance(end, date) else parse_compact_date(end)
    if d0 > d1:
        raise DateError(f"start {d0} after end {d1}")
    days = (d1 - d0).days + (1 if inclusive else 0)
    return [format_compact_date(d0 + timedelta(days=i)) for i in range(days)]


def shift_compact_date(value: str, days: int) -> str:
    """Shift a compact date by N days (negative allowed)."""
    return format_compact_date(parse_compact_date(value) + timedelta(days=days))


def nearest_valid_date(valid: list[str] | tuple[str, ...],
                       target: str | None = None) -> str | None:
    """The valid date nearest to ``target`` (default: today ET), newest-first tiebreak.

    ``valid`` may be in any order; it is sorted descending internally.
    Returns None when ``valid`` is empty. Mirrors the reference frontend's
    "sport reset / first visit lands on the nearest date with games" rule.
    """
    if not valid:
        return None
    ordered = sorted(set(valid), reverse=True)
    if target is None:
        return ordered[0]
    t = parse_compact_date(target)
    # Tiebreak on distance first, then NEWEST date wins (forward-looking:
    # prefer the latest snapshot when two dates are equally near).
    best = min(ordered,
               key=lambda d: (abs(parse_compact_date(d) - t), -parse_compact_date(d).toordinal()))
    return best


def iso_to_eastern_display(iso_utc: str) -> str:
    """Format an ISO-8601 UTC timestamp as a short ET display string.

    Returns "" for empty/invalid input (callers render a fallback). Example
    output: ``Sep 07, 7:05 PM ET``.
    """
    if not iso_utc:
        return ""
    try:
        ts = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    local = ts.astimezone(ET)
    # %d is zero-padded; strip the leading zero like the reference UI.
    day = str(local.day)
    return (f"{local.strftime('%b')} {day}, "
            f"{local.strftime('%I:%M %p ET').lstrip('0')}")


def is_evening_start_et(iso_utc: str) -> bool:
    """True when the game starts at 7 PM ET or later (day/night tag rule)."""
    if not iso_utc:
        return False
    try:
        ts = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(ET).hour >= 19


def artifact_date_from_filename(name: str, prefix: str, ext: str) -> str | None:
    """Extract the ``YYYYMMDD`` stamp from ``<prefix>_<date><ext>``.

    Returns None when the name does not match the family pattern or the
    stamp is not an 8-digit date. SHAP-style names with a middle segment
    (``shap_game_20260907_WSH@SD.csv``) resolve their stamp from the first
    8-digit run after the prefix.
    """
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    if rest.startswith("_"):
        rest = rest[1:]
    if ext:
        suffix = ext if ext.startswith(".") else f".{ext}"
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
        elif rest.endswith(ext):
            rest = rest[: -len(ext)]
    core = rest.split("_")[0] if "_" in rest else rest
    if len(core) == DATE_LEN and core.isdigit():
        try:
            parse_compact_date(core)
        except DateError:
            return None
        return core
    return None
