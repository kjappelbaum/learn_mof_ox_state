# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import concurrent.futures
import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)
from tqdm import tqdm
from six.moves import range
import time


def _bootstrap_metric_fold(_, model, X, y, scoring_funcs, sample_idx, rng):  # pylint:disable=too-many-arguments
    scores = {}
    bootstrap_idx = rng.choice(sample_idx, size=sample_idx.shape[0], replace=True)
    prediction = model.predict(X[bootstrap_idx])
    for metricname, metric in scoring_funcs:
        scores[metricname] = metric(y[bootstrap_idx], prediction)
    return scores


def bootstrapped_metrics(  # pylint:disable=too-many-arguments
        model, X, y, scoring_funcs, n_rounds=200, seed=1234, max_workers=6) -> list:
    """Get bootstrapped statistics for the metrics estimated with the callables in scoring_funcs

    Arguments:
        model {sklearn model} -- sklearn model that needs to be tests
        X {np.array} -- array with features
        y {np.array} -- array with labels
        scoring_funcs {list} -- list of tuples (name, callable), where the callable is a scoring function
            that takes y_hat and y and returns a float

    Keyword Arguments:
        n_rounds {int} -- number of bootstrap samples (default: {200})
        seed {int} -- random seed (default: {1234})


    Returns:
        list -- list of dictionaries of metrics
    """
    rng = np.random.RandomState(seed)

    sample_idx = np.arange(X.shape[0])

    bootstrap_fold_partial = partial(
        _bootstrap_metric_fold,
        model=model,
        X=X,
        y=y,
        scoring_funcs=scoring_funcs,
        sample_idx=sample_idx,
        rng=rng,
    )
    metrics = []
    rounds = list(range(n_rounds))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for metric in tqdm(executor.map(bootstrap_fold_partial, list(rounds)), total=len(rounds)):
            metrics.append(metric)

    return metrics


def return_scoring_funcs():
    """

    Returns:
        list -- list of tuples (name, scoring function)
    """
    f1_micro = partial(f1_score, average='micro')
    f1_macro = partial(f1_score, average='macro')
    precision = partial(precision_score, average='micro')
    recall = partial(recall_score, average='micro')
    metrics = [
        ('accuracy', accuracy_score),
        ('balanced_accuracy', balanced_accuracy_score),
        ('f1_micro', f1_micro),
        ('f1_macro', f1_macro),
        ('precision', precision),
        ('recall', recall),
    ]

    return metrics


def get_metrics_dict(true, predicted, dummy=False):
    accuracy = accuracy_score(true, predicted)
    f1_micro = f1_score(true, predicted, average='micro')
    f1_macro = f1_score(true, predicted, average='macro')
    balanced_accuracy = balanced_accuracy_score(true, predicted)
    precision = precision_score(true, predicted, average='micro')
    recall = recall_score(true, predicted, average='micro')

    return {
        'points': len(predicted),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'dummy': dummy,
    }


def _permutation_score_base(_, model, X, y, cv, metric_func, shuffle=True):  # pylint:disable=too-many-arguments, too-many-locals
    model_ = model
    avg_score = []
    if shuffle:  # shuffle modifies in place!
        # np.random.shuffle(X)
        np.random.shuffle(y)

    print('entering CV loop')
    count = 0
    for train, test in cv.split(X, y):
        print(('CV iteration {}'.format(count)))
        X_train = X[train]
        X_test = X[test]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = y[train]
        y_test = y[test]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3)
        model_.fit(X_train, y_train)
        model_._calibrate_base_estimators('isotonic', X_valid, y_valid)  # pylint:disable=protected-access
        avg_score.append(metric_func(y_test, model_.predict(X_test)))
        count += 1

        print((np.mean(np.array(avg_score))))
    return np.mean(np.array(avg_score))


def permutation_test(model, X, y, rounds=30, metric_func=balanced_accuracy_score, max_workers=12):  # pylint:disable=too-many-arguments,unused-argument
    cv = StratifiedKFold(10)
    permuted_scores = []
    model_ = model
    score = _permutation_score_base(None, model, X, y, cv, metric_func, shuffle=False)
    base_permutation_score = partial(_permutation_score_base, model=model_, X=X, y=y, cv=cv, metric_func=metric_func)

    print('*** Now starting the shuffling ***')
    rounds = list(range(rounds))
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for metric in tqdm(executor.map(base_permutation_score, list(rounds)), total=len(rounds)):
            print(metric)
            np.save('permuted_metric_{}'.format(time.strftime('%Y%m%d-%H%M%S')), metric)
            permuted_scores.append(metric)

    permuted_scores = np.array(permuted_scores)
    p_value = (np.sum(permuted_scores >= score) + 1.0) / (len(rounds) + 1)
    return score, permuted_scores, p_value
