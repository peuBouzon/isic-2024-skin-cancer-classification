from pathlib import Path

RAW_METADATA_PATH = Path('/home/pedro/ufes/datasets/ISIC2024/train-metadata.csv')
ONE_HOT_ENCODED_PATH = Path('/home/pedro/ufes/datasets/ISIC2024/isic-2024-one-hot-encoded.csv')


# Gradient Boosting Decision Trees
GBDT_BEST_HYPERPARAMETERS_FILE = Path('./gbdt_best_hyperparameters.json')

# Naive Bayes
NAIVE_BAYES_BEST_FEATURES_FILE = Path('./nb_best_features.json')