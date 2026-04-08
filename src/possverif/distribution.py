"""Core possibilistic distribution and verification scorecard.

Terminology and equations follow:
  "Possible, Yes; Ignorant, Perhaps: A Scorecard for Possibilistic Forecasts"
  (Lawson et al., in prep for J. Atmos. Sci.)

Notation mapping:
  π       raw possibility distribution, values in [0, 1]
  π'      normalized distribution: π'(ω) = π(ω) / m
  m       commitment (peak possibility): max_ω π(ω)
  H_Π     ignorance: 1 - m
  α*      depth-of-truth: π'(c_obs)
  η       diffuseness: (1/K) Σ π'(c)
  δ       support margin: α* - η
  N_c*    conditional necessity of truth: 1 - max_{ω ≠ c_obs} π'(ω)
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple


class Scorecard(NamedTuple):
    """Five-number post-event verification scorecard."""

    depth_of_truth: float         # α* ∈ [0, 1]
    diffuseness: float            # η  ∈ [1/K, 1]
    support_margin: float         # δ  ∈ [-(1-1/K), 1-1/K]
    ignorance: float              # H_Π ∈ [0, 1)
    conditional_necessity: float  # N_c* ∈ [0, 1]


class PossibilisticDistribution:
    """A possibilistic forecast distribution over categorical outcomes.

    Parameters
    ----------
    possibilities : dict[str, float]
        Mapping of category names to raw possibility values in [0, 1].
        At least one value must be > 0.

    Examples
    --------
    >>> dist = PossibilisticDistribution({"bkg": 0.2, "mod": 0.8, "elv": 0.3, "ext": 0.0})
    >>> dist.commitment
    0.8
    >>> dist.ignorance
    0.19999999999999996
    >>> dist.scorecard("mod").depth_of_truth
    1.0
    """

    def __init__(self, possibilities: Dict[str, float]) -> None:
        for k, v in possibilities.items():
            if not 0 <= v <= 1:
                raise ValueError(
                    f"Possibility for '{k}' must be in [0, 1], got {v}"
                )

        self.raw = dict(possibilities)
        self.categories = list(possibilities.keys())
        self.K = len(self.categories)

        # Commitment: peak possibility value (m)
        self.commitment = max(possibilities.values())
        if self.commitment == 0:
            raise ValueError("At least one possibility must be > 0")

        # Ignorance: unassigned fraction of plausibility budget (H_Π)
        self.ignorance = 1.0 - self.commitment

        # Normalized (shape) distribution: π' = π / m
        self.normalized = {
            k: v / self.commitment for k, v in possibilities.items()
        }

    # ------------------------------------------------------------------
    # Pre-event metrics (no observation required)
    # ------------------------------------------------------------------

    def diffuseness(self) -> float:
        """Diffuseness η = (1/K) Σ π'(c).

        Ranges from 1/K (all mass in one category) to 1 (uniform).
        """
        return sum(self.normalized.values()) / self.K

    def necessity(self, event: str) -> float:
        """Necessity N(A) = 1 - max_{ω ∉ A} π(ω)."""
        others = [v for k, v in self.raw.items() if k != event]
        if not others:
            return 1.0
        return 1.0 - max(others)

    def conditional_necessity(self, event: str) -> float:
        """Conditional necessity N_c(A) = 1 - max_{ω ≠ A} π'(ω)."""
        others = [v for k, v in self.normalized.items() if k != event]
        if not others:
            return 1.0
        return 1.0 - max(others)

    def entropy(self) -> float:
        """Possibilistic (Hartley-style) entropy.

        Computed from the sorted raw distribution:
            H = Σ_j (π_{j-1} - π_j) · log2(j-1)
        where π values are sorted descending with 0 appended.
        """
        sorted_vals = sorted(self.raw.values(), reverse=True)
        if sorted_vals[-1] != 0:
            sorted_vals.append(0)

        h = 0.0
        for j in range(1, len(sorted_vals)):
            diff = sorted_vals[j - 1] - sorted_vals[j]
            if j > 1:
                h += diff * math.log2(j - 1)
        return h

    # ------------------------------------------------------------------
    # Post-event scorecard
    # ------------------------------------------------------------------

    def scorecard(self, observed: str) -> Scorecard:
        """Compute the five-number verification scorecard.

        Parameters
        ----------
        observed : str
            The category that was actually observed.

        Returns
        -------
        Scorecard
            Named tuple with (depth_of_truth, diffuseness, support_margin,
            ignorance, conditional_necessity).
        """
        if observed not in self.raw:
            raise ValueError(
                f"'{observed}' not in categories: {self.categories}"
            )

        alpha_star = self.normalized[observed]
        eta = self.diffuseness()
        delta = alpha_star - eta
        h_pi = self.ignorance
        n_c_star = self.conditional_necessity(observed)

        return Scorecard(
            depth_of_truth=alpha_star,
            diffuseness=eta,
            support_margin=delta,
            ignorance=h_pi,
            conditional_necessity=n_c_star,
        )

    # ------------------------------------------------------------------
    # Pignistic bridge: possibility → probability
    # ------------------------------------------------------------------

    def to_probability(self) -> Dict[str, float]:
        """Convert to a (K+1)-dimensional probability vector.

        Reserves ignorance as an explicit outcome, then distributes
        remaining mass proportionally among categories.

        Returns
        -------
        dict[str, float]
            Probability for each category plus ``"_ignorance"``.
            Sums to 1.
        """
        total_poss = sum(self.raw.values())
        remaining = 1.0 - self.ignorance  # = commitment

        probs = {
            k: (v * remaining) / total_poss
            for k, v in self.raw.items()
        }
        probs["_ignorance"] = self.ignorance
        return probs

    def surprise(self, observed: str, floor: float = 0.01) -> float:
        """Log score: S = -log2(p(c_obs)).

        Uses the pignistic-bridge probability, floored at ``floor``
        to prevent infinite surprise when p = 0.

        Parameters
        ----------
        observed : str
            The observed category.
        floor : float
            Minimum probability before taking the log (default 0.01).
        """
        probs = self.to_probability()
        p = max(probs[observed], floor)
        return -math.log2(p)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cats = ", ".join(f"{k}={v:.2f}" for k, v in self.raw.items())
        return (
            f"PossibilisticDistribution({cats}, "
            f"m={self.commitment:.2f}, H_Π={self.ignorance:.2f})"
        )
