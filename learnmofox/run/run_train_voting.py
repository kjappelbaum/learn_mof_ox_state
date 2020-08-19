from joblib import load, dump
from comet_ml import Experiment
import os
import time
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


STARTTIMESTRING = time.strftime("%Y%m%d-%H%M%S")

FEAT_TRAIN_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/features_train.npy"
FEAT_TEST_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/features_test.npy"
FEAT_VALID_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/features_valid.npy"

LABEL_TRAIN_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/labels_train.npy"
LABEL_TEST_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/labels_test.npy"
LABEL_VALID_PATH = "/scratch/kjablonk/oximachine_all/merged_dataset2/labels_valid.npy"


from sklearn.preprocessing import StandardScaler

from learnmofox.utils import VotingClassifier

experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY", None), project_name="mof-oxidation-states",
)

X_train = np.load(FEAT_TRAIN_PATH)
X_valid = np.load(FEAT_VALID_PATH)
X_test = np.load(FEAT_TEST_PATH)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

vt = VarianceThreshold(0.1)
X_train = vt.fit_transform(X_train)
X_valid = vt.transform(X_valid)
X_test = vt.transform(X_test)

y_train = np.load(LABEL_TRAIN_PATH)
y_valid = np.load(LABEL_VALID_PATH)
y_test = np.load(LABEL_TEST_PATH)


def model_eval(
    models: list,
    xtrain: np.array,
    ytrain: np.array,
    xtest: np.array,
    ytest: np.array,
    postfix: str = "0",
    outdir_metrics: str = None,
    outdir_models: str = None,
):
    """Peforms a model evaluation on training and test set and dump the predictions with the actual values
    into an outout file

    Arguments:
        models {list} -- list of tuples with model name and model itself
        xtrain {np.array} -- feature matrix training set
        ytrain {np.array} -- label vector training set
        xtest {np.array} -- feature matrix test set
        ytest {np.array} -- label vector test set
        postfix {str} -- string that will be attached to filename
        outdir_metrics {str} -- output directory for metrics
        outdir_models {str} -- output directory for models
    """
    predictions = []

    for name, model in models:
        outdir_metrics_verbose = os.path.join(os.path.join(outdir_metrics, "verbose"))
        if not os.path.exists(outdir_metrics_verbose):
            os.mkdir(outdir_metrics_verbose)

        outname_base_metrics = os.path.join(
            outdir_metrics_verbose, "_".join([STARTTIMESTRING, name, postfix])
        )
        outname_base_models = os.path.join(
            outdir_models, "_".join([STARTTIMESTRING, name, postfix])
        )

        train_true = ytrain
        test_true = ytest

        train_predict = model.predict(xtrain)
        test_predict = model.predict(xtest)
        accuracy_train = accuracy_score(train_true, train_predict)
        accuracy_test = accuracy_score(test_true, test_predict)
        f1_micro_train = f1_score(train_true, train_predict, average="micro")
        f1_micro_test = f1_score(test_true, test_predict, average="micro")

        f1_macro_train = f1_score(train_true, train_predict, average="macro")
        f1_macro_test = f1_score(test_true, test_predict, average="macro")

        balanced_accuracy_train = balanced_accuracy_score(train_true, train_predict)
        balanced_accuracy_test = balanced_accuracy_score(test_true, test_predict)
        precision_train = precision_score(train_true, train_predict, average="micro")
        precision_test = precision_score(train_true, train_predict, average="micro")
        recall_train = recall_score(train_true, train_predict, average="micro")
        recall_test = recall_score(test_true, test_predict, average="micro")

        prediction = {
            "model": name,
            "postfix": postfix,
            "outname_base_models": outname_base_models,
            "outname_base_metrics": outname_base_metrics,
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "f1_micro_train": f1_micro_train,
            "f1_micro_test": f1_micro_test,
            "f1_macro_train": f1_macro_train,
            "f1_macro_test": f1_macro_test,
            "balanced_accuracy_train": balanced_accuracy_train,
            "balanced_accuracy_test": balanced_accuracy_test,
            "precision_train": precision_train,
            "precision_test": precision_test,
            "recall_train": recall_train,
            "recall_test": recall_test,
            "training_points": len(ytrain),
            "test_points": len(ytest),
        }

        arrays = {
            "train_true": train_true,
            "train_predict": train_predict,
            "test_predict": test_predict,
            "test_true": test_true,
        }

        arrays.update(prediction)

        predictions.append(arrays)

        with open(outname_base_metrics + ".pkl", "wb") as fh:
            pickle.dump(arrays, fh)

        dump(model, outname_base_models + ".joblib")


optimized_models = [
    ("gradient_boosting", load("models/20200814-184230_gradient_boosting_0.joblib")),
    ("sgd", load("models/20200814-184230_sgd_0.joblib")),
    ("knn", load("models/20200814-184230_knn_0.joblib")),
    ("extra_trees", load("models/20200814-184230_extra_trees_0.joblib")),
]

vc = VotingClassifier(optimized_models, voting="soft")

vc._calibrate_base_estimators("sigmoid", X_valid, y_valid)

model_eval(
    [("ensemble", vc)],
    X_train,
    y_train,
    X_test,
    y_test,
    outdir_metrics="metrics",
    outdir_models="models",
)
