import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils.params import load_params


def eval(data_path, model_path):

    # load test data
    X_test = pd.read_csv(f"{data_path}/X_train.csv", index_col=None)
    y_test = pd.read_csv(f"{data_path}/y_train.csv", index_col=None)

    # load model and get predictions
    model = load(model_path)
    y_pred = model.predict(X_test)

    # calculate and store the scores
    metrics = {
        "recal": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    json.dump(obj=metrics, fp=open("metrics.json", "w"), indent=4, sort_keys=True)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)

    eval(
        data_path=params.base.data_processed_path,
        model_path=params.train.model_path,
    )
