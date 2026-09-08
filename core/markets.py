"""Market settlement framework with TIE/PUSH distinction and configurable
integer settlement.

Two distinct concepts, never conflated:

* **TIE** — the *game* (or the game's regulation period) ends with no winner.
  A property of the sport/scope, not of a line. NFL full-game ties exist;
  NHL *regulation* ties exist while the full-game moneyline is resolved by
  OT/SO; MLB/NBA never tie (extras/OT).
* **PUSH** — a *wager* is refunded because the final result lands exactly on
  an integer line. A property of a market offer, not of the game.

``settlement_scope`` selects which score the settlement reads:
``regulation`` (the score at the end of regulation) or ``full_game``
(including OT/SO). A market's ability to tie or push is configured per
sport/market via :class:`SettlementConfig` — never hard-coded to totals
only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from core.config import normalize_sport_key


class Outcome(enum.Enum):
    HOME = "home"
    AWAY = "away"
    TIE = "tie"
    PUSH = "push"


class SettlementError(ValueError):
    """Raised when a market cannot be settled from the given inputs."""


# ---------------------------------------------------------------------------
# Settlement configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementConfig:
    """Whether tie/push outcomes are possible for one market type.

    ``settlement_scope`` selects the score the settlement reads:

    * ``regulation`` — the score at the end of regulation (NFL: 60 minutes;
      NHL: 60 minutes; MLB: 9 innings; NBA: 48 minutes).
    * ``full_game`` — the final score including OT/SO/extras.

    ``tie_allowed`` gates the TIE outcome (game-level no-winner result).
    ``push_allowed`` gates the PUSH outcome (wager refunded on an integer
    line). A half-point line can never push regardless of this flag.
    """

    settlement_scope: str = "full_game"
    tie_allowed: bool = False
    push_allowed: bool = True

    def __post_init__(self) -> None:
        if self.settlement_scope not in ("regulation", "full_game"):
            raise SettlementError(
                f"settlement_scope must be 'regulation' or 'full_game', "
                f"got {self.settlement_scope!r}")


# Prebuilt configs for the sport-specific cases the platform serves.
REGULATION_TIE_ALLOWED = SettlementConfig(
    settlement_scope="regulation", tie_allowed=True, push_allowed=True)
FULL_GAME_TIE_ALLOWED = SettlementConfig(
    settlement_scope="full_game", tie_allowed=True, push_allowed=True)
FULL_GAME_NO_TIE = SettlementConfig(
    settlement_scope="full_game", tie_allowed=False, push_allowed=True)


# ---------------------------------------------------------------------------
# Settlement result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settlement:
    """The settled result of one offer against a final score."""

    outcome: Outcome
    line: float
    margin: float  # signed: positive = home side of the line
    is_decided: bool
    scope: str = "full_game"

    @property
    def is_push(self) -> bool:
        # Undecided settlements carry Outcome.PUSH as a placeholder; a
        # real push is a decided refund of an integer-line wager.
        return self.outcome is Outcome.PUSH and self.is_decided

    @property
    def is_tie(self) -> bool:
        return self.outcome is Outcome.TIE


def _settle_margin(margin: float, line: float, cfg: SettlementConfig,
                   *, decided: bool) -> Settlement:
    """Resolve a signed margin against a line under the given config."""
    if not decided:
        return Settlement(Outcome.PUSH, line, 0.0, is_decided=False,
                          scope=cfg.settlement_scope)
    if margin > 0:
        out = Outcome.HOME
    elif margin < 0:
        out = Outcome.AWAY
    else:
        out = Outcome.TIE if cfg.tie_allowed else Outcome.PUSH
    return Settlement(out, line, margin, is_decided=True,
                      scope=cfg.settlement_scope)


def _push_check(margin: float, line: float, cfg: SettlementConfig) -> None:
    """A half-point line can never push; integer lines push only if allowed."""
    if margin == 0 and not float(line).is_integer():
        raise SettlementError(
            f"half-point line {line} cannot push or tie "
            f"(margin {margin})")
    if margin == 0 and not cfg.push_allowed and not cfg.tie_allowed:
        raise SettlementError(
            f"exact landing on line {line} but neither push nor tie is "
            f"allowed by the settlement config")


# ---------------------------------------------------------------------------
# Scalar line settlement (totals and spread/run lines)
# ---------------------------------------------------------------------------


def settle_total(total_line: float, final_total: float | None,
                 cfg: SettlementConfig = FULL_GAME_NO_TIE) -> Settlement:
    """Settle a totals offer under a settlement config.

    ``Outcome.HOME`` means the OVER side won, ``Outcome.AWAY`` the UNDER
    side (the convention every totals monitor in the platform shares).
    A push requires an integer line; a half-point line with an exact
    landing is a data error, never a silent win. Totals cannot tie (the
    game must produce a total), so ``cfg.tie_allowed`` is ignored here
    except through ``_push_check``.
    """
    if final_total is None:
        return Settlement(Outcome.PUSH, total_line, 0.0, is_decided=False,
                          scope=cfg.settlement_scope)
    margin = float(final_total) - float(total_line)
    if margin != 0:
        _push_check(margin, total_line, cfg)  # no-op for non-zero margins
        out = Outcome.HOME if margin > 0 else Outcome.AWAY
        return Settlement(out, total_line, margin, is_decided=True,
                          scope=cfg.settlement_scope)
    # Exact landing: push if allowed and the line is integer; a half-point
    # line landing exactly is a data error.
    if not float(total_line).is_integer():
        raise SettlementError(
            f"half-point line {total_line} cannot push (total {final_total})")
    if not cfg.push_allowed:
        raise SettlementError(
            f"total {final_total} lands exactly on integer line "
            f"{total_line} but push is not allowed by the settlement config")
    return Settlement(Outcome.PUSH, total_line, margin, is_decided=True,
                      scope=cfg.settlement_scope)


def settle_spread(line: float, home_margin: float | None,
                  *, home_side: bool = True,
                  cfg: SettlementConfig = FULL_GAME_NO_TIE) -> Settlement:
    """Settle a spread/run-line offer for one side under a config.

    ``line`` is the handicap applied to the chosen side, signed from that
    side's perspective: a favorite giving 1.5 passes ``line=-1.5`` with
    ``home_side=True``; a dog taking 1.5 passes ``line=+1.5`` with
    ``home_side=False``. The side covers when ``side_margin + line > 0``.

    ``Outcome.HOME``/``Outcome.AWAY`` mean "the offered side won/lost"
    regardless of which dugout it is. Integer lines push on exact landing
    when ``push_allowed``; half-point lines never push — a level margin
    against a ±0.5 line simply goes to the +0.5 side (cover/lose, no
    push). A game-level tie does NOT refund a spread wager: a 24-24 NFL
    final against -3 settles as a failure to cover for the favorite
    (standard book rule) — TIE is a moneyline/outright concept only.
    """
    if home_margin is None:
        return Settlement(Outcome.PUSH, line, 0.0, is_decided=False,
                          scope=cfg.settlement_scope)
    side_margin = float(home_margin) if home_side else -float(home_margin)
    margin = side_margin + line
    if margin > 0:
        out = Outcome.HOME
    elif margin < 0:
        out = Outcome.AWAY
    else:
        # Exact landing on the line.
        if not float(line).is_integer():
            raise SettlementError(
                f"half-point line {line} cannot push (margin {side_margin})")
        if not cfg.push_allowed:
            raise SettlementError(
                f"margin {side_margin} lands exactly on integer line "
                f"{line} but push is not allowed by the settlement config")
        out = Outcome.PUSH
    return Settlement(out, line, margin, is_decided=True,
                      scope=cfg.settlement_scope)


def settle_moneyline(home_score: float | None,
                     away_score: float | None,
                     cfg: SettlementConfig = FULL_GAME_NO_TIE) -> Settlement:
    """Settle a moneyline under a settlement config.

    With ``tie_allowed=True`` (NFL full-game, NHL regulation) a level
    score settles TIE — the game has no winner, and the moneyline wager
    is resolved as a tie (refund-equivalent), NOT a line push.

    With ``tie_allowed=False`` (MLB/NBA full game, NHL full game after
    OT/SO) a level score is *undecided* (``is_decided=False``): the data
    shows no resolution, so no winner is fabricated. This is a data
    anomaly for these sports, never a silent TIE/PUSH.
    """
    if home_score is None or away_score is None:
        return Settlement(Outcome.PUSH, 0.0, 0.0, is_decided=False,
                          scope=cfg.settlement_scope)
    margin = float(home_score) - float(away_score)
    if margin > 0:
        return Settlement(Outcome.HOME, 0.0, margin, is_decided=True,
                          scope=cfg.settlement_scope)
    if margin < 0:
        return Settlement(Outcome.AWAY, 0.0, margin, is_decided=True,
                          scope=cfg.settlement_scope)
    if cfg.tie_allowed:
        return Settlement(Outcome.TIE, 0.0, margin, is_decided=True,
                          scope=cfg.settlement_scope)
    # No-tie sport, level score: undecided, never fabricated.
    return Settlement(Outcome.PUSH, 0.0, margin, is_decided=False,
                      scope=cfg.settlement_scope)


# ---------------------------------------------------------------------------
# Probability triples (over/push/under, cover/push/lose, home/tie/away)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbTriple:
    """A probability triple over a line: (side_a, push, side_b)."""

    p_a: float
    p_push: float
    p_b: float

    def __post_init__(self) -> None:
        vals = (self.p_a, self.p_push, self.p_b)
        for v in vals:
            if not 0.0 <= v <= 1.0:
                raise SettlementError(f"probability {v!r} outside [0,1]")
        total = sum(vals)
        if abs(total - 1.0) > 1e-6:
            raise SettlementError(
                f"probabilities sum to {total}, expected 1.0")

    def decided_mass(self) -> float:
        return self.p_a + self.p_b

    def renormalized_pair(self) -> tuple[float, float]:
        """The two-way (a, b) probabilities with push mass removed.

        When there is no decided mass the pair is the honest (0.5, 0.5)
        coin flip — the reference UI's coin-flip pill, not a fabricated
        edge.
        """
        d = self.decided_mass()
        if d <= 0.0:
            return 0.5, 0.5
        return self.p_a / d, self.p_b / d


@dataclass(frozen=True)
class TieTriple:
    """A three-way probability triple: (home, tie, away).

    Used for sports/scopes where the game itself can tie (NFL full-game,
    NHL regulation). ``p_tie`` is first-class mass — never forced to null
    on a three-way regulation market.
    """

    p_home: float
    p_tie: float
    p_away: float

    def __post_init__(self) -> None:
        vals = (self.p_home, self.p_tie, self.p_away)
        for v in vals:
            if not 0.0 <= v <= 1.0:
                raise SettlementError(f"probability {v!r} outside [0,1]")
        total = sum(vals)
        if abs(total - 1.0) > 1e-6:
            raise SettlementError(
                f"probabilities sum to {total}, expected 1.0")

    def decided_mass(self) -> float:
        return self.p_home + self.p_away

    def renormalized_pair(self) -> tuple[float, float]:
        """Two-way (home, away) with tie mass removed (honest coin flip at 0)."""
        d = self.decided_mass()
        if d <= 0.0:
            return 0.5, 0.5
        return self.p_home / d, self.p_away / d


def normalize_push_probs(
    p_a: float,
    p_b: float,
    p_push: float | None = None,
) -> tuple[float, float, float]:
    """Normalize a (p_a, p_b[, p_push]) triple to sum exactly to 1.

    With ``p_push=None`` the push mass is derived as ``1 - p_a - p_b``
    (clamped at 0). Raises on negative inputs or a derived push below
    -1e-9. Returns the triple rounded to 12 decimals for deterministic
    serialization.
    """
    if p_a < 0 or p_b < 0:
        raise SettlementError("probabilities must be non-negative")
    if p_push is None:
        push = 1.0 - p_a - p_b
        if push < -1e-9:
            raise SettlementError(
                f"p_a + p_b = {p_a + p_b} exceeds 1; no push mass left")
        push = max(push, 0.0)
    else:
        if p_push < 0:
            raise SettlementError("p_push must be non-negative")
        push = p_push
    total = p_a + p_b + push
    if total <= 0:
        raise SettlementError("total probability mass is zero")
    a, b, p = p_a / total, p_b / total, push / total
    return round(a, 12), round(b, 12), round(p, 12)


def normalize_tie_probs(p_home: float, p_away: float,
                        p_tie: float | None = None,
                        ) -> tuple[float, float, float]:
    """Tie-aware three-way normalization returning (p_home, p_away, p_tie).

    Same contract as :func:`normalize_push_probs` but the third mass is a
    game-level TIE (no winner), not a wager push. With ``p_tie=None`` the
    tie mass is derived as ``1 - p_home - p_away``. The tie mass is
    first-class: a three-way regulation market keeps its explicit tie
    probability — it is never forced to null/zero by normalization.
    """
    return normalize_push_probs(p_home, p_away, p_tie)


def triple_from_line_grid(
    p_over: float, p_under: float, p_push: float | None = None,
) -> ProbTriple:
    """Build a validated ProbTriple from an artifact's over/under pair."""
    a, b, p = normalize_push_probs(p_over, p_under, p_push)
    return ProbTriple(p_a=a, p_push=p, p_b=b)


def push_display_bucket(p_push: float) -> str:
    """The UI bucket for push mass (honesty labels, never hidden).

    Buckets match the reference dashboard's push display conventions.
    """
    if p_push < 0.0:
        raise SettlementError("p_push must be non-negative")
    if p_push < 0.005:
        return "none"
    if p_push < 0.05:
        return "small"
    if p_push < 0.15:
        return "moderate"
    return "high"


def sport_allows_ties(sport: str) -> bool:
    """Whether the sport's *full-game* result can tie (drives tie pills).

    MLB: no (extras). NBA: no (OT). NHL: no (OT/SO resolves the moneyline).
    NFL: yes (rare full-game ties). NHL *regulation* ties are handled by
    passing ``REGULATION_TIE_ALLOWED`` to the settlement functions — the
    full-game moneyline is always decided.
    """
    return normalize_sport_key(sport) == "nfl"
