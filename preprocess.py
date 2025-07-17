import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from dataset import ISIC2024

def fill_nan(train:pd.DataFrame, test:pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train[ISIC2024.NUMERICAL_FEATURES])
    train[ISIC2024.NUMERICAL_FEATURES] = imputer.transform(train[ISIC2024.NUMERICAL_FEATURES])
    test[ISIC2024.NUMERICAL_FEATURES] = imputer.transform(test[ISIC2024.NUMERICAL_FEATURES])
    return train, test

if __name__ == "__main__":
    df = pd.read_csv('./train-metadata.csv')

    df = pd.get_dummies(df, columns=ISIC2024.RAW_CATEGORICAL_FEATURES, dtype=int)
    
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    df[ISIC2024.FOLDER_COLUMN] = None
    for i, (_, test_indexes) in enumerate(kfold.split(df, df[ISIC2024.TARGET_COLUMN], groups=df[ISIC2024.PATIENT_ID])):
        df.loc[test_indexes, ISIC2024.FOLDER_COLUMN] = i + 1

    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/isic-2024-one-hot-encoded.csv', index=False)