# Hybrid zero detector — minimal public starter

This repository contains a **minimal public implementation of the core detector layer**
used in the manuscript on hybrid Gaussian–wavelet sampled-data zero localization.

## Current contents

- `detector.py`  
  Core detector implementation based on:
  - Gaussian smoothing of sampled data,
  - a smoothed derivative proxy,
  - the practical detector score  
    \[
    D = \frac{|P|}{|W| + \varepsilon},
    \]
  - candidate extraction from local minima of the detector score,
  - optional Newton refinement helper.

- `example_usage.py`  
  Minimal runnable example on a synthetic oscillatory test function.

## Scope of this starter release

This public starter is intentionally limited to the **detector core** and a **minimal usage example**.
It is **not** a full reproduction package for all manuscript experiments.

In particular, the following are **not yet included** in this minimal release:

- the full Sturm–Liouville benchmark scripts,
- the noisy-\Xi benchmark scripts,
- the clustered-zero resolution experiments,
- the complete table/figure regeneration pipeline.

These experiment-specific scripts are being consolidated and cleaned for a later public release.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Minimal usage example

Run:

```bash
python example_usage.py
```

The script will:
1. generate a sampled oscillatory function,
2. run the hybrid detector,
3. print candidate basin locations,
4. display a figure with the sampled function and detector score.

## Notes

This starter release is intended to support **method-level reproducibility** of the detector core.
The full experimental pipelines used for the manuscript will be released after final code consolidation and cleanup.
