import json
import torch
import optuna
import numpy as np
import torch.nn.functional as F

from dataset import ISIC2024
from metrics import get_partial_auc
from torch.utils.data import DataLoader
from naivebayes import NaiveBayesEnsemble

RANDOM_STATE = 42
BATCH_SIZE = 128

def objective(trial):
    categorical_suggestions = {ft: trial.suggest_categorical(ft, [True, False]) for ft in ISIC2024.RAW_CATEGORICAL_FEATURES}
    numerical_suggestions = {ft: trial.suggest_categorical(ft, [True, False]) for ft in ISIC2024.NUMERICAL_FEATURES}

    all_preds = []
    all_labels = []
    for folder in range(1, 6):
      train_dataset = ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=folder, return_tensors=True, balance=False)
      class_weights = len(train_dataset) / torch.bincount(train_dataset.labels)
      test_dataloader = DataLoader(ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=folder, train=False, return_tensors=True), 
                                  batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=4, pin_memory=True, 
                                  persistent_workers=True, prefetch_factor=2)

      numerical_features = [train_dataset.get_indexes(ft) for ft, keep in numerical_suggestions.items() if keep]
      if len(numerical_features) > 0:
          numerical_features = np.concatenate(numerical_features)
      else:
          numerical_features = np.array([])
      categorical_features = [train_dataset.get_indexes(ft) for ft, keep in categorical_suggestions.items() if keep]
      if len(categorical_features) > 0:
          categorical_features = np.concatenate(categorical_features)
      else:
          categorical_features = np.array([])
      model = NaiveBayesEnsemble(n_estimators=1, sampling_strategy=0.01, random_state=RANDOM_STATE)
      model.fit(train_dataset.metadata, train_dataset.lbs, 
                categorical_features, numerical_features,
                weights=class_weights)

      for batch, labels in test_dataloader:
          all_preds.append(F.softmax(model(batch), dim=1)[:, 1])
          all_labels.append(labels)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    return get_partial_auc(preds.numpy(), labels.numpy(), min_tpr=0.80)

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', study_name='nb_feature_selection', sampler=optuna.samplers.GPSampler())
    study.optimize(objective, n_trials=250, n_jobs=1)

    print("Best pAUC:", study.best_value)
    print('Salvando best_hyperparameters.json...')
    with open('nb_best_features_fix.json', 'w') as f:
      json.dump(study.best_params, f, indent=4)
