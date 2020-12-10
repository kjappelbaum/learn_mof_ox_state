# -*- coding: utf-8 -*-
import os
import pickle
import sys

from joblib import load

from learnmofox import utils

THIS_DIR = os.path.dirname(__file__)

sys.modules["utils"] = utils

model = load(
    os.path.join(THIS_DIR, "..", "examples", "votingclassifier_patched.joblib")
)
scaler = load(os.path.join(THIS_DIR, "..", "examples", "scaler_0.joblib"))

with open(os.path.join(THIS_DIR, "..", "examples", "features.pkl"), "rb") as fh:
    features_dict = pickle.load(fh)


def test_prediction():
    expected = [
        [2, 2],
        [2, 2],
        [4, 4],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],
        [2, 2],
        [2, 2, 2, 2],
    ]
    for i, k in enumerate(features_dict.keys()):
        X = scaler.transform(features_dict[k])
        prediction = model.predict(X)
        assert (prediction == expected[i]).all()
