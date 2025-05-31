import os
from typing import Tuple

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Download HomeCredit_columns_description.csv
    if not os.path.exists(config.DATASET_DESCRIPTION):
        gdown.download(
            config.DATASET_DESCRIPTION_URL, config.DATASET_DESCRIPTION, quiet=False
        )

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    app_train = pd.read_csv(config.DATASET_TRAIN)
    app_test = pd.read_csv(config.DATASET_TEST)
    columns_description = pd.read_csv(config.DATASET_DESCRIPTION_URL)

    return app_train, app_test, columns_description


def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    X_train =  app_train.drop(columns='TARGET')
    y_train = app_train['TARGET']
    X_test = app_test.drop(columns='TARGET')
    y_test = app_test['TARGET']



    return X_train, y_train, X_test, y_test


def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
 
    return X_train, X_val, y_train, y_val
