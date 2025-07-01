import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from dataset import ISIC2024

df = pd.read_csv('./train-metadata.csv')

for ft in ISIC2024.categorical_features:
    is_nan = df[ft].isna() | df[ft].isin(['Unknown'])
    if is_nan.sum() > 0:
        print(f'Filling missing values for {ft} with mode "{df[ft].mode().values[0]}"...')
        df.loc[is_nan, ft] = df[ft].mode()

for ft in ISIC2024.numerical_features:
    is_nan = df[ft].isna()
    if is_nan.sum() > 0:
        print(f'Filling missing values for {ft} with mean "{df[ft].mean()}"...')
        df.loc[is_nan, ft] = df[ft].mean()

df = pd.get_dummies(df, columns=ISIC2024.categorical_features, dtype=int)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['folder'] = None
for i, (_, test_indexes) in enumerate(kfold.split(df, df['target'])):
    df.loc[test_indexes, 'folder'] = i + 1

os.makedirs('./data', exist_ok=True)
df.to_csv('./data/isic-2024-one-hot-encoded.csv', index=False)