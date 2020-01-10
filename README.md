# learn_mof_ox_state

[![Actions Status](https://github.com/kjappelbaum/learn_mof_ox_state/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/learn_mof_ox_state/actions)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![DOI](https://zenodo.org/badge/208837592.svg)](https://zenodo.org/badge/latestdoi/208837592)
[![Maintainability](https://api.codeclimate.com/v1/badges/2a0c417e69517a2738d2/maintainability)](https://codeclimate.com/github/kjappelbaum/learn_mof_ox_state/maintainability)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Tools to train and test a voting classifier that predicts oxidation states (of MOFs).

> ⚠️ **Warning**: You need to export `COMET_API_KEY`, as the code will look for it if you want to track your experiments. If you do not want to do this, remove those lines in the code.

> ⚠️ **Warning**: Some parts of the code requires some modifications in the packages, for which I did not make PRs so far, you need to use my forks. You need `pip>=18.1` for this to be set up automatically. More details can be found below.

**Info**: For some parts, there are also [`.dvc` files](https://dvc.org) that allow to reproduce the pipeline.

## Installation

To install the software with all dependencies, you can use

```bash
pip install git+https://github.com/kjappelbaum/learn_mof_ox_state.git
```

This should, for appropriate versions of pip (`pip>=18.1`), also install [our fork of matminer from the correct branch](https://github.com/kjappelbaum/matminer.git@localpropertystats).
This automatically installs several command-line tools (CLI) which are detailed below.

The full process should take some seconds.

## Usage

- The functions in this package requires input (features and labels) that can be generated with our [mine_mof_ox python package]().
  The full datasets which can be used to train a model, as well as a pre-trained model are deposited on the [MaterialsCloud Archive (doi: 10.24435/materialscloud:2019.0085/v1 )](https://doi.org/10.24435/materialscloud:2019.0085/v1). The analysis command line interfaces can be used to reproduce our findings, based on the data deposited in the MaterialsCloud Archive. The training CLI can for example be used as

```bash
  python machine_learn_oxstates/learnmofox/train_ensemble_classifier.py {featurespath} {labelspath} {modelpath} {metricsoutpath} standard soft isotonic 40000 20 none --train_one_fold
```

- Some of the experiments we ran, together with code and datahash, can also be found at [comet.ml](https://www.comet.ml/kjappelbaum/mof-oxidation-states/view/)

- For testing a pre-trained model we recommend using our [webapp](https://dev-tools.materialscloud.org/oximachine/input_structure/), for which the code can be found, along with the Docker images, in another [Github repository](https://github.com/kjappelbaum/oximachinetool).

## File contents

### Training

The training can, depending on the training set size, take hours.

- `train_calibrate_voting_classifier_no_track.py`: to run the training without comet.ml
- `train_calibrate_voting_classifier.py`: train a voting classifier (with optimized hyperparameters and track the experiments with comet.ml)
- `train_ensemble_classifier.py`: run the hyperparameter optimization for the ensemble of models
- `utils.py`: contains the custom voting classifier class and some utils

### Analysis

The runtime for the tests depends on whether they require retraining the model (permutation significance), which can take several hours, or whether they only involve evaluating the model for some data points, which will take minutes.

- `feature_importance_cli.py`: command-line-tools to calculate feature importance with permutation or SHAP
- `farm_learning_curves.py`: command-line-tool to run learning curves
- `bias_variance_cli.py`: run a bias-variance decomposition analysis with mlxtend
- `permutation_significance.py`: tool to run a permutation significance test (permute label and measure metrics to see if the model learned something meaningful)
- `run_combinatorial_study.py`: train models on different feature subsets
- `metrics.py` contains helper functions to calculate metrics
- `bootstrapped_metrics.py`: functions to calculate a bootstrapped learning curve point
- `test_model.py`: command-line-tool to run some basic tests
