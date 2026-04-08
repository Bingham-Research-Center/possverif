"""Information-theoretic metrics via the pignistic bridge.

Provides Brier score, log score (surprise), KL divergence, and
mutual information operating on PossibilisticDistribution objects
through the probability conversion in distribution.to_probability().
"""

from __future__ import annotations

import math
from typing import Dict

from possverif.distribution import PossibilisticDistribution


def brier_score(dist: PossibilisticDistribution, observed: str) -> float:
    """Brier score for a possibilistic forecast after probability conversion.

    BS = Σ_c (p_c - o_c)^2  where o_c = 1 if c == observed, else 0.

    Lower is better. Includes the ignorance outcome in the sum.
    """
    probs = dist.to_probability()
    bs = 0.0
    for cat, p in probs.items():
        indicator = 1.0 if cat == observed else 0.0
        bs += (p - indicator) ** 2
    return bs


def log_score(dist: PossibilisticDistribution, observed: str,
              floor: float = 0.01) -> float:
    """Log score (surprise): S = -log2(p(c_obs)).

    Alias for dist.surprise(observed, floor). Provided here for
    a consistent information-module API.
    """
    return dist.surprise(observed, floor=floor)


def kl_divergence(dist: PossibilisticDistribution,
                  reference: Dict[str, float]) -> float:
    """KL divergence D_KL(p_dist || p_ref) in bits.

    Both distributions are over the same category set. The dist is
    converted to probability via the pignistic bridge; the reference
    is already a probability dict (e.g., climatology).

    Categories in dist but not in reference are skipped.
    A floor of 1e-10 prevents log(0).
    """
    probs = dist.to_probability()
    dkl = 0.0
    for cat, p in probs.items():
        q = reference.get(cat, 1e-10)
        if p > 0:
            dkl += p * math.log2(max(p, 1e-10) / max(q, 1e-10))
    return dkl


def information_gain(dist: PossibilisticDistribution,
                     baseline: Dict[str, float],
                     observed: str,
                     floor: float = 0.01) -> float:
    """Information gain relative to a baseline forecast.

    IG = S(baseline, c_obs) - S(dist, c_obs)

    Positive IG means dist is more informative than baseline.
    Both are evaluated as log scores on the observed category.
    """
    p_base = max(baseline.get(observed, floor), floor)
    s_base = -math.log2(p_base)
    s_dist = dist.surprise(observed, floor=floor)
    return s_base - s_dist
