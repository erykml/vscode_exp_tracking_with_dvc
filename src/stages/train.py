import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils.params import load_params


def train(data_path, model_type, model_path, random_state, **train_params):

    # load training data
    X_train = pd.read_csv(f"{data_path}/X_train.csv", index_col=None)
    y_train = pd.read_csv(f"{data_path}/y_train.csv", index_col=None)

    # pick the model
    if model_type == "rf":
        model = RandomForestClassifier(random_state=random_state, **train_params)
    elif model_type == "lgbm":
        model = LGBMClassifier(random_state=random_state, **train_params)
    else:
        raise ValueError("Unsupported model_type!")

    # trains the model and store it
    model.fit(X_train, y_train)
    dump(model, model_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)

    model_dir = Path(params.train.model_dir)
    model_dir.mkdir(exist_ok=True)

    train(
        data_path=params.base.data_processed_path,
        model_type=params.train.model_type,
        model_path=params.train.model_path,
        random_state=params.base.random_state,
        **params.train.params,
    )
