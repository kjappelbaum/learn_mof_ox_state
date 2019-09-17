# learn_mof_ox_state

 > ⚠️ **Warning**:  You need to export `COMET_API_KEY` as the code will look for it


## Training

* run every step of the pipeline from collection of the features to the analysis using DVC

* run 10-fold stratified cross-validation


## Feature selection

* Drop maybe one class of feature vectors


## Analysis

* Run test with bootstrapped metrics on the special test sets

* Run [randomization test](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html#sphx-glr-auto-examples-feature-selection-plot-permutation-test-for-classification-py), cf. sOjala, M.; Garriga, G. C. Permutation Tests for Studying Classifier Performance. In 2009 Ninth IEEE International Conference on Data Mining; IEEE: Miami Beach, FL, USA, 2009; pp 908–913. https://doi.org/10.1109/ICDM.2009.108.
