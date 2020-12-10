# learn_mof_ox_state

[![Actions Status](https://github.com/kjappelbaum/learn_mof_ox_state/workflows/Python%20package/badge.svg)](https://github.com/kjappelbaum/learn_mof_ox_state/actions)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/download/releases/3.6.0/)
[![DOI](https://zenodo.org/badge/208837592.svg)](https://zenodo.org/badge/latestdoi/208837592)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kjappelbaum/learn_mof_ox_state/master?filepath=examples%2Fexample.ipynb)

Tools to train and test a voting classifier that predicts oxidation states (of MOFs).

> ⚠️ **Warning**: You need to export `COMET_API_KEY`, as the code will look for it if you want to track your experiments (when you retrain the model). If you do not want to do this, remove those lines in the code. You might also want to consider other tracking options such as weights and biases.

## Installation

To install the software with all dependencies, you can use

```bash
pip install git+https://github.com/kjappelbaum/learn_mof_ox_state.git
```

The full process should take some seconds.

Note that the models have been fitted using `scikit-learn==0.21.3` and therefore one should ideally used this version. For better compatibility with the other dependencies (`matminer`, `apricot`) that depend on newer versions of `scikit-learn` we patched the model by adding the `_strategy` attribute to the initialization `DummyClassifier` of the `GradientBoostingClassifier` and adding the `n_samples_fit_` attribute to the `KNeighborsClassifier`. If you plan to do some further developments, it might be advisable to bump all dependencies before training a new model.

## Usage

- The functions in this package requires inputs (features and labels) that can be generated with our [oximachine_featurizer Python package](https://github.com/kjappelbaum/mof_oxidation_states).
  The full datasets which can be used to train a model, as well as a pre-trained model are deposited on the [MaterialsCloud Archive (doi: 10.24435/materialscloud:2019.0085/v1 )](https://doi.org/10.24435/materialscloud:2019.0085/v1). The analysis command line interfaces can be used to reproduce our findings, based on the data deposited in the MaterialsCloud Archive. The training CLI can for example be used as

```bash
  python machine_learn_oxstates/learnmofox/train_ensemble_classifier.py {featurespath} {labelspath} {modelpath} {metricsoutpath} standard soft isotonic 40000 20 none --train_one_fold
```

- Some experiments we ran, together with code and datahash, can also be found at [comet.ml](https://www.comet.ml/kjappelbaum/mof-oxidation-states/view/)

- For testing a pre-trained model we recommend using our [webapp](go.epfl.ch/oximachine), for which the code can be found, along with the Docker images, in another [GitHub repository](http://github.com/kjappelbaum/oximachinetool). There is also a small Python package, [oximachinerunner](https://github.com/kjappelbaum/oximachinerunner), that allows to run inference on crystal structures.

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

## Example usage

The use of the main functions of this package is shown in the Jupyter Notebook in the example directory.
It contains some example structures and the output, which should be produces in seconds.
