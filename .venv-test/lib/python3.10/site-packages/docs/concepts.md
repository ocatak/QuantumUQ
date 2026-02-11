## Concepts: Uncertainty in Quantum Machine Learning

Quantum models introduce several sources of uncertainty:

- **Shot noise (statistical)**: finite sampling of measurement outcomes.
- **Hardware noise (aleatoric)**: decoherence, gate errors, readout noise.
- **Model uncertainty (epistemic)**: limited training data, model misspecification.

QuantumUQ focuses on *model-agnostic* techniques that sit on top of existing QML models:

- **ShotBootstrap**: resample shots / repeated forward passes.
- **DeepEnsemble**: independent predictors trained from different initializations or data splits.
- **NoiseProfile**: sweep shots and quantify stability (entropy and probability variance).

All methods work on a small **Predictor protocol** (`predict`, `predict_proba`, and `task`).

