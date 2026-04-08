"""Tests against the paper's three worked examples (Section 07).

Scenario A: Sharp-correct MDT forecast
Scenario B: Hedged-correct ENH forecast
Scenario C: Sharp-wrong NONE forecast

Expected values from the paper's Table ref:tab:scorecard_comparison.
"""

import math
import pytest

from possverif import (
    PossibilisticDistribution, Scorecard,
    brier_score, log_score, information_gain,
    aggregate_scorecards,
)


# --- Paper scenarios (SPC categories) ---

SCENARIO_A = PossibilisticDistribution({
    "NONE": 0.0, "MRGL": 0.0, "SLGT": 0.05,
    "ENH": 0.15, "MDT": 0.90, "HIGH": 0.10,
})

SCENARIO_B = PossibilisticDistribution({
    "NONE": 0.1, "MRGL": 0.1, "SLGT": 0.4,
    "ENH": 0.55, "MDT": 0.3, "HIGH": 0.0,
})

SCENARIO_C = PossibilisticDistribution({
    "NONE": 0.85, "MRGL": 0.1, "SLGT": 0.05,
    "ENH": 0.0, "MDT": 0.0, "HIGH": 0.0,
})


class TestPreEventMetrics:
    def test_commitment(self):
        assert SCENARIO_A.commitment == 0.90
        assert SCENARIO_B.commitment == 0.55
        assert SCENARIO_C.commitment == 0.85

    def test_ignorance(self):
        assert abs(SCENARIO_A.ignorance - 0.10) < 1e-10
        assert abs(SCENARIO_B.ignorance - 0.45) < 1e-10
        assert abs(SCENARIO_C.ignorance - 0.15) < 1e-10

    def test_normalized_peak_is_one(self):
        assert SCENARIO_A.normalized["MDT"] == 1.0
        assert SCENARIO_B.normalized["ENH"] == 1.0
        assert SCENARIO_C.normalized["NONE"] == 1.0


class TestScorecard:
    """Five-number scorecard from paper's Table."""

    def test_scenario_a_mdt_observed(self):
        sc = SCENARIO_A.scorecard("MDT")
        assert abs(sc.depth_of_truth - 1.0) < 1e-10
        assert abs(sc.ignorance - 0.10) < 1e-10
        # η = mean of normalized: (0 + 0 + 0.0556 + 0.1667 + 1.0 + 0.1111) / 6
        assert abs(sc.diffuseness - 0.2222) < 0.001
        assert abs(sc.support_margin - 0.7778) < 0.001
        # N_c* = 1 - max(others normalized) = 1 - 0.1667
        assert abs(sc.conditional_necessity - 0.8333) < 0.001

    def test_scenario_b_enh_observed(self):
        sc = SCENARIO_B.scorecard("ENH")
        assert abs(sc.depth_of_truth - 1.0) < 1e-10
        assert abs(sc.ignorance - 0.45) < 1e-10
        # Normalized: NONE=0.182, MRGL=0.182, SLGT=0.727, ENH=1.0, MDT=0.545, HIGH=0
        # η = (0.182 + 0.182 + 0.727 + 1.0 + 0.545 + 0) / 6 = 0.439
        assert abs(sc.diffuseness - 0.439) < 0.002
        assert abs(sc.support_margin - 0.561) < 0.002
        # N_c* = 1 - max(others norm) = 1 - 0.727
        assert abs(sc.conditional_necessity - 0.273) < 0.002

    def test_scenario_c_mdt_observed(self):
        sc = SCENARIO_C.scorecard("MDT")
        # MDT has raw possibility 0.0, so normalized = 0.0
        assert abs(sc.depth_of_truth - 0.0) < 1e-10
        assert abs(sc.ignorance - 0.15) < 1e-10
        # Normalized: NONE=1.0, MRGL=0.1176, SLGT=0.0588, ENH=0, MDT=0, HIGH=0
        # η = (1.0 + 0.1176 + 0.0588 + 0 + 0 + 0) / 6 = 0.196
        assert abs(sc.diffuseness - 0.196) < 0.002
        assert abs(sc.support_margin - (-0.196)) < 0.002
        # N_c* = 1 - max(others norm) = 1 - 1.0 = 0
        assert abs(sc.conditional_necessity - 0.0) < 1e-10

    def test_sharp_correct_beats_hedged(self):
        """Scenario A should have higher support margin than B."""
        sc_a = SCENARIO_A.scorecard("MDT")
        sc_b = SCENARIO_B.scorecard("ENH")
        assert sc_a.support_margin > sc_b.support_margin

    def test_wrong_forecast_negative_support(self):
        """Scenario C (wrong) should have negative support margin."""
        sc_c = SCENARIO_C.scorecard("MDT")
        assert sc_c.support_margin < 0


class TestPignisticBridge:
    def test_probabilities_sum_to_one(self):
        probs = SCENARIO_A.to_probability()
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_ignorance_outcome_present(self):
        probs = SCENARIO_A.to_probability()
        assert "_ignorance" in probs
        assert abs(probs["_ignorance"] - 0.10) < 1e-10

    def test_peak_category_gets_most_probability(self):
        probs = SCENARIO_A.to_probability()
        cat_probs = {k: v for k, v in probs.items() if k != "_ignorance"}
        assert max(cat_probs, key=cat_probs.get) == "MDT"


class TestInformation:
    def test_brier_perfect_is_low(self):
        # A sharp correct forecast should have low Brier
        bs_a = brier_score(SCENARIO_A, "MDT")
        bs_c = brier_score(SCENARIO_C, "MDT")
        assert bs_a < bs_c

    def test_surprise_correct_vs_wrong(self):
        # Surprise for correct outcome should be lower
        s_a = log_score(SCENARIO_A, "MDT")
        s_c = log_score(SCENARIO_C, "MDT")
        assert s_a < s_c

    def test_information_gain_positive_for_good_forecast(self):
        # Uniform baseline
        baseline = {cat: 1/6 for cat in SCENARIO_A.categories}
        baseline["_ignorance"] = 0.0
        ig = information_gain(SCENARIO_A, baseline, "MDT")
        assert ig > 0


class TestAggregation:
    def test_aggregate_scorecards(self):
        scs = [
            SCENARIO_A.scorecard("MDT"),
            SCENARIO_B.scorecard("ENH"),
        ]
        agg = aggregate_scorecards(scs)
        assert "depth_of_truth" in agg
        assert agg["depth_of_truth"]["n"] == 2
        assert agg["depth_of_truth"]["mean"] == 1.0  # both are 1.0


class TestEdgeCases:
    def test_all_zero_except_one(self):
        dist = PossibilisticDistribution({"a": 1.0, "b": 0.0})
        assert dist.commitment == 1.0
        assert dist.ignorance == 0.0
        sc = dist.scorecard("a")
        assert sc.depth_of_truth == 1.0
        assert sc.conditional_necessity == 1.0

    def test_all_equal(self):
        dist = PossibilisticDistribution({"a": 0.5, "b": 0.5, "c": 0.5})
        assert dist.commitment == 0.5
        assert dist.ignorance == 0.5
        assert abs(dist.diffuseness() - 1.0) < 1e-10  # all normalized to 1

    def test_raises_on_all_zero(self):
        with pytest.raises(ValueError):
            PossibilisticDistribution({"a": 0.0, "b": 0.0})

    def test_raises_on_out_of_range(self):
        with pytest.raises(ValueError):
            PossibilisticDistribution({"a": 1.5})
