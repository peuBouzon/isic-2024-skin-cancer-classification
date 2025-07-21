from pathlib import Path

RAW_METADATA_PATH = Path('/home/pedro/ufes/datasets/ISIC2024/train-metadata.csv')
ONE_HOT_ENCODED_PATH = Path('/home/pedro/ufes/datasets/ISIC2024/isic-2024-one-hot-encoded.csv')

# Gradient Boosting Decision Trees
GBDT_BEST_HYPERPARAMETERS_FILE = Path('./gbdt_best_hyperparameters.json')

# Naive Bayes
NAIVE_BAYES_BEST_FEATURES_FILE = Path('./nb_best_features.json')

# É necessário informar o mesmo número de resultados para cada método.
EXPERIMENT_DIRS_BY_METHOD = {
  'mlp' : ['20250719_051429', '20250719_134715', '20250719_181157'],
  'gbdt': ['20250719_083226', '20250720_083315', '20250720_084828'],
  'naivebayes': ['20250719_133536', '20250720_085734', '20250720_085818'],
}
