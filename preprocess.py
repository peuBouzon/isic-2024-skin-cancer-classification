import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from dataset import ISIC2024
import polars as pl
import numpy as np
import config

def fill_nan(train:pd.DataFrame, test:pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train[ISIC2024.NUMERICAL_FEATURES])
    train[ISIC2024.NUMERICAL_FEATURES] = imputer.transform(train[ISIC2024.NUMERICAL_FEATURES])
    test[ISIC2024.NUMERICAL_FEATURES] = imputer.transform(test[ISIC2024.NUMERICAL_FEATURES])
    return train, test

if __name__ == "__main__":
    df = (pl.read_csv(config.RAW_METADATA_PATH)
            .with_columns(pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64))
            .with_columns(pl.col(ISIC2024.RAW_CATEGORICAL_FEATURES).cast(pl.Categorical),
        )
        .to_pandas())

    df = pd.get_dummies(df, columns=ISIC2024.RAW_CATEGORICAL_FEATURES, dtype=int)
    
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    df[ISIC2024.FOLDER_COLUMN] = None
    for i, (_, test_indexes) in enumerate(kfold.split(df, df[ISIC2024.TARGET_COLUMN], groups=df[ISIC2024.PATIENT_ID])):
        df.loc[test_indexes, ISIC2024.FOLDER_COLUMN] = i + 1

    df.to_csv(config.ONE_HOT_ENCODED_PATH, index=False)