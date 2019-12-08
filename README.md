# learn_mof_ox_state

[![Actions Status](https://github.com/kjappelbaum/learn_mof_ox_state/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/learn_mof_ox_state/actions)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![DOI](https://zenodo.org/badge/208837592.svg)](https://zenodo.org/badge/latestdoi/208837592)

Tools to train and test a voting classifier that predicts oxidation states (of MOFs).

> ⚠️ **Warning**: You need to export `COMET_API_KEY`, as the code will look for it if you want to track your experiments. If you do not want to do this, remove those lines in the code.

> ⚠️ **Warning**: Some parts of the code requires some modifications in the packages, for which I did not make PRs so far, you need to use my forks. You need `pip>=18.1` for this to be set up automatically

> ⚠️ **Info**: For some parts, there are also `.dvc` files that allow to reproduce the pipeline.

## Training

- `train_calibrate_voting_classifier_no_track.py`: to run the training without comet.ml
- `train_calibrate_voting_classifier.py`: train a voting classifier (with optimized hyperparameters and track the experiments with comet.ml)
- `train_ensemble_classifier.py`: run the hyperparameter optimization for the ensemble of models
- `utils.py`: contains the custom voting classifier class and some utils

## Analysis

- `feature_importance_cli.py`: command-line-tools to calculate feature importance with permutation or SHAP
- `farm_learning_curves.py`: command-line-tool to run learning curves
- `bias_variance_cli.py`: run a bias-variance decomposition analysis with mlxtend
- `permutation_significance.py`: tool to run a permutation significance test (permute label and measure metrics to see if the model learned something meaningful)
- `run_combinatorial_study.py`: train models on different feature subsets
- `metrics.py` contains helper functions to calculate metrics
- `bootstrapped_metrics.py`: functions to calculate a bootstrapped learning curve point
- `test_model.py`: command-line-tool to run some basic tests
