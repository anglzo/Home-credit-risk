from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import lightgbm as lgb

def fix_anomalous_days( train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    return working_train_df, working_val_df, working_test_df


#Pipeline building 

def build_pipeline(train_df: pd.DataFrame) -> tuple[Pipeline, list]:

    numeric_cols = [row for row in train_df.columns if train_df[row].dtype != 'object']
    binary_cols = [row for row in train_df.columns if train_df[row].dtype == 'object' and train_df[row].nunique() == 2]
    categ_cols = [row for row in train_df.columns if train_df[row].dtype == 'object' and row not in binary_cols]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    multiclass_pipeline = Pipeline([
        ('imputer' , SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocesor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('bin', binary_pipeline, binary_cols),
        ('multi', multiclass_pipeline, categ_cols)
    ])

    params = {
    'task' : 'train',
    'objective' : 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.1,
    'verbose': -1,
    'n_jobs': -1
}
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocesor),
        ('model', lgb.LGBMClassifier(**params))
    ])

    return pipeline, numeric_cols + binary_cols + categ_cols