# -*- coding: utf-8 -*-
import os
import pickle
from glob import glob
from pathlib import Path

import joblib
import numpy as np

from learnmofox import metrics
from learnmofox.utils import read_pickle


def evaluate_model(
    modelpath, featurespath, labelspath, namespath, scalerpath
):  # pylint:disable=too-many-locals
    X = np.load(featurespath)
    y = np.load(labelspath)
    names = read_pickle(namespath)
    classifier = joblib.load(modelpath)
    scaler = joblib.load(scalerpath)
    X = scaler.transform(X)

    predicted = classifier.predict(X)
    results = metrics.get_metrics_dict(y, predicted)

    points = np.arange(0, len(y))

    diffs = [
        (names[point], i, j)
        for point, i, j in list(zip(points, predicted, y))
        if i != j
    ]

    p = Path(modelpath)
    modelstem = p.stem

    results["diffs"] = diffs
    results["modelpath"] = modelpath
    results["modelname"] = modelstem

    folder = p.parts[-2].replace("model_", "")

    results["features"] = folder

    with open(os.path.join("results_{}_{}.pkl".format(folder, modelstem)), "wb") as fh:
        pickle.dump(results, fh)

    return results


def main():  # pylint:disable=too-many-locals
    model_ensemble = glob(os.path.join("*", "*_ensemble*.joblib"))
    models_et = glob(os.path.join("*", "*extra_trees*.joblib"))
    models_gb = glob(os.path.join("*", "*gb*.joblib"))
    models_knn = glob(os.path.join("*", "*knn*.joblib"))
    models_sgd = glob(os.path.join("*", "*sgd*.joblib"))

    models = model_ensemble + models_et + models_gb + models_knn + models_sgd

    result_list = []

    for model in models:
        try:
            p = Path(model)
            folder = p.parts[-2]
            basename = folder.replace("model_", "")
            holdoutpath = "_".join(["houldout", basename])
            featurespath = os.path.join(holdoutpath, "features.npy")
            labelspath = os.path.join(holdoutpath, "labels.npy")
            namespath = os.path.join(holdoutpath, "names.pkl")
            date = p.stem.split("-")[0]

            scalerpath = glob(os.path.join(folder, "{}-*scaler*.joblib".format(date)))[
                0
            ]
            result_list.append(
                evaluate_model(model, featurespath, labelspath, namespath, scalerpath)
            )
        except Exception:
            pass

    with open(os.path.join("results_eval.pkl"), "wb") as fh:
        pickle.dump(result_list, fh)


if __name__ == "__main__":
    main()
