---
title: Calibration in Quantum Machine Learning
description: >-
  Expected calibration error, Brier score, negative log-likelihood, and
  reliability diagrams for quantum classifiers, and why more measurement
  shots do not by themselves fix a miscalibrated model.
---

# Calibration in Quantum Machine Learning

A classifier's *confidence* and its *correctness* are two different
quantities, and a model can be wrong about the relationship between them in
either direction. Calibration is the study of whether they actually match:
of the predictions a model calls "90% confident," is roughly 90% of that
group actually correct?

## Confidence vs. correctness

An **overconfident** model reports higher confidence than its accuracy
supports -- it is wrong more often than its own numbers suggest, which is
the dangerous failure mode for anything downstream that trusts the
confidence score. An **underconfident** model is the opposite: correct more
often than it claims, which is safer but wastes information. A
well-calibrated model's confidence is an honest estimate of its accuracy at
that confidence level, in either direction.

## Metrics QuantumUQ measures

- **Expected Calibration Error (`ece`)**: bins predictions by confidence and
  averages, weighted by bin size, the gap between each bin's confidence and
  its actual accuracy. Zero means perfectly calibrated on the evaluated
  data; it says nothing about accuracy itself.
- **Brier score (`brier`)**: mean squared error between predicted class
  probabilities and the one-hot true label -- a single number combining
  calibration and accuracy for classification.
- **Negative log-likelihood (`nll`)**: the average log-loss of the true
  class's predicted probability; like Brier score, it reflects both
  calibration and accuracy, and penalizes confident wrong answers heavily.
- **Predictive entropy**: how spread out a prediction's class probabilities
  are, independent of whether those probabilities are correct. High entropy
  means "the model itself is unsure"; low entropy means confident, whether
  or not that confidence is warranted.
- **Reliability diagrams**: a plot of accuracy against confidence, binned as
  in `ece`, against the diagonal line of perfect calibration. QuantumUQ
  doesn't ship a plotting function for this, but every notebook under
  `examples/notebooks/` that discusses calibration builds one directly from
  `PredictiveDistribution.mean` in a few lines -- see
  `examples/notebooks/05_ece_calibration_bugfix.ipynb` for a worked example.

## Measurement uncertainty is not calibration

It's tempting to assume that a noisier measurement (fewer shots) means a
less trustworthy prediction, and that running more shots would therefore
improve calibration. That conflates two separate things:

- **Measurement uncertainty** is how much a single prediction would change
  if you re-ran the circuit with the same shot budget -- this is what
  `ShotBootstrap`'s `dist.std` reports, and it does shrink as shot count
  increases (see
  [Shot Noise and Finite-Shot Uncertainty](qml_shot_noise.md)).
- **Calibration** is whether the model's *average* confidence, over many
  different inputs, matches its *average* accuracy. That relationship is
  set by training -- the model architecture, the training data, the loss
  function -- not by how precisely you measure any one prediction
  afterward.

A model that is miscalibrated at 100 shots is generally still miscalibrated
at 10,000 shots: more shots make the confidence estimate itself more
precise, but they don't change what that confidence is *of*. The
[PennyLane Community Demo notebook](https://github.com/ocatak/QuantumUQ/blob/main/examples/notebooks/08_pennylane_community_demo.ipynb)
measures this directly on a trained variational classifier: sweeping shots
from 100 to 10,000 shrinks the `ShotBootstrap` uncertainty by roughly the
expected `1/sqrt(shots)` factor, while accuracy and ECE stay close to flat
across the same range. Do not assume the opposite relationship holds for
every model and dataset -- verify it, the same way that notebook does,
rather than treating "more shots" as a calibration fix.

## What QuantumUQ does, and does not, do

QuantumUQ **measures** calibration -- `ece`, `nll`, `brier`, and predictive
entropy tell you whether and how badly a model is miscalibrated. It does
**not** currently provide an automatic calibration-correction step (such as
temperature scaling or isotonic regression); fixing a calibration problem
once you've found one is on the roadmap, not yet implemented. If `ece`
reports a large gap, that's a diagnosis, not something QuantumUQ has
already corrected for you.

## Getting started

- [Try QuantumUQ in 15 Minutes](fifteen_minutes.md) -- computing `ece` and
  predictive entropy on a real prediction.
- [Uncertainty quantification in quantum machine learning](qml_uncertainty_quantification.md) --
  how calibration relates to the other uncertainty sources QuantumUQ
  measures.
- [Shot noise and finite-shot uncertainty](qml_shot_noise.md) -- the
  measurement-uncertainty side of this distinction.
- [Reliability benchmarking](qml_reliability_benchmarking.md) -- tracking
  ECE, Brier score, and NLL across a shot-count sweep automatically.
