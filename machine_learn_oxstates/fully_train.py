# -*- coding: utf-8 -*-
"""
This module loads a model where the hyperparameters were already optimized using
hyperopt and trains it on the full dataset. Leave still a holdout set aside
to evaluate this model.
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
from hyperopt import tpe, anneal, rand, mix
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from utils import VotingClassifier
from mlxtend.evaluate import BootstrapOutOfBag
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from ml_insights import SplineCalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import pandas as pd
from joblib import dump, load
import concurrent.futures
import click

RANDOM_SEED = 1234
STARTTIMESTRING = time.strftime('%Y%m%d-%H%M%S')
MIN_SAMPLES = 10


def load_model(modelpath):
    model = load(modelpath)

def train_base_models():
    ...
