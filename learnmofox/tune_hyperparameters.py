# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments, too-many-locals, line-too-long, logging-fstring-interpolation

"""
This is to replace the "train_ensemble_classifier" module
which is overly complicated
"""

from __future__ import absolute_import
from __future__ import print_function
from functools import partial
import time
from collections import Counter
import numpy as np
import os
import json
import pickle
import logging
from typing import Tuple
from comet_ml import Experiment
from hyperopt import tpe, anneal, rand, mix, hp
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from learnmofox.utils import VotingClassifier
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
from joblib import dump
import concurrent.futures
import click

RANDOM_SEED = 1234
STARTTIMESTRING = time.strftime('%Y%m%d-%H%M%S')
MIN_SAMPLES = 10
classifiers = [
    (
        'sgd',
        partial(
            components.sgd,
            loss=hp.pchoice('loss', [(0.5, 'log'), (0.5, 'modified_huber')]),
        ),
    ),
    ('knn', components.knn),
    ('gradient_boosting', partial(components.gradient_boosting, loss='deviance')),
    ('extra_trees', components.extra_trees),
    ('svr', components.svc_rbf),
    ('nb', components.gaussian_nb),
]

trainlogger = logging.getLogger('trainer')
trainlogger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(filename)s: %(message)s')
filehandler = logging.FileHandler(os.path.join('logs', STARTTIMESTRING + '_train.log'))
filehandler.setFormatter(formatter)
trainlogger.addHandler(filehandler)