# possverif

Python implementation of the possibilistic forecast verification scorecard from:

> "Possible, Yes; Ignorant, Perhaps: A Scorecard for Possibilistic Forecasts"
> (Lawson et al., in prep for J. Atmos. Sci.)

## Install

```bash
pip install -e .
```

## Quick start

```python
from possverif import PossibilisticDistribution

# Create a forecast distribution
dist = PossibilisticDistribution({
    "bkg": 0.2, "mod": 0.8, "elv": 0.3, "ext": 0.0
})

# Pre-event properties
dist.commitment      # m = 0.8  (peak possibility)
dist.ignorance       # H_Π = 0.2  (unassigned plausibility)
dist.diffuseness()   # η  (spread of normalized shape)
dist.entropy()       # Hartley-style possibilistic entropy

# Post-event five-number scorecard
sc = dist.scorecard("mod")
sc.depth_of_truth       # α*   Did truth get peak support?
sc.diffuseness          # η    How spread was the forecast?
sc.support_margin       # δ    Excess support for truth vs average
sc.ignorance            # H_Π  System confidence
sc.conditional_necessity  # N_c*  Dominance of truth over runner-up

# Pignistic bridge (possibility → probability)
probs = dist.to_probability()  # (K+1) vector with explicit ignorance
dist.surprise("mod")           # -log2(p(c_obs)), in bits
```
