"""Sample-level aggregation of possibilistic verification metrics.

Computes mean, std, and confidence intervals of scorecard metrics
and information scores across a verification sample.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from possverif.distribution import PossibilisticDistribution, Scorecard


def aggregate_scorecards(scorecards: Sequence[Scorecard]) -> Dict[str, float]:
    """Compute mean of each scorecard metric across a sample.

    Parameters
    ----------
    scorecards : sequence of Scorecard
        One scorecard per forecast-observation pair.

    Returns
    -------
    dict
        Keys are metric names (e.g., 'depth_of_truth'), values are
        dicts with 'mean', 'std', 'n'.
    """
    fields = Scorecard._fields
    arrays = {f: np.array([getattr(sc, f) for sc in scorecards])
              for f in fields}
    return {
        f: {'mean': float(arr.mean()),
            'std': float(arr.std()),
            'n': len(arr)}
        for f, arr in arrays.items()
    }


def aggregate_forecasts(
    distributions: Sequence[PossibilisticDistribution],
    observations: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute aggregate scorecard from forecast-observation pairs.

    Parameters
    ----------
    distributions : sequence of PossibilisticDistribution
    observations : sequence of str
        Observed category for each forecast.

    Returns
    -------
    dict
        Aggregated scorecard statistics.
    """
    scorecards = [d.scorecard(obs) for d, obs in zip(distributions, observations)]
    return aggregate_scorecards(scorecards)
