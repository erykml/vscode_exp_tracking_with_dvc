import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.params import load_params


def split_data(
    data_path,
    data_file,
    processed_path,
    target_col,
    test_size,
    exclude_cols,
    random_state=42,
):
    """
    Function used for splitting data intro training and validation sets.
    """

    df = pd.read_csv(os.path.join(data_path, data_file))

    X = df.drop(columns=exclude_cols).copy()
    y = X.pop(target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train.to_csv(f"{processed_path}/X_train.csv", index=None)
    X_test.to_csv(f"{processed_path}/X_test.csv", index=None)
    y_train.to_csv(f"{processed_path}/y_train.csv", index=None)
    y_test.to_csv(f"{processed_path}/y_test.csv", index=None)


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)

    data_processed_path = Path(params.base.data_processed_path)
    data_processed_path.mkdir(exist_ok=True)

    split_data(
        data_path=params.base.data_path,
        data_file=params.base.data_file,
        processed_path=params.base.data_processed_path,
        target_col=params.base.target_col,
        test_size=params.data_split.test_size,
        exclude_cols=params.base.exclude_cols,
        random_state=params.base.random_state,
    )
