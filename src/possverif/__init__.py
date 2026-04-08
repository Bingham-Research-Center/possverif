"""Possibilistic forecast verification scorecard.

Implements the verification framework from:
  "Possible, Yes; Ignorant, Perhaps: A Scorecard for Possibilistic Forecasts"

Usage::

    from possverif import PossibilisticDistribution

    dist = PossibilisticDistribution({"bkg": 0.2, "mod": 0.8, "elv": 0.3, "ext": 0.0})
    dist.commitment      # 0.8
    dist.ignorance       # 0.2
    dist.diffuseness()   # mean of normalized values

    sc = dist.scorecard("mod")
    sc.depth_of_truth    # α*
    sc.support_margin    # δ
"""

from possverif.distribution import PossibilisticDistribution, Scorecard

__all__ = ["PossibilisticDistribution", "Scorecard"]
