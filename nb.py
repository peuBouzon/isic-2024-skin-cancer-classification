import json
import torch
import numpy as np
import torch.nn.functional as F

from dataset import ISIC2024
from naivebayes import NaiveBayesEnsemble
from torch.utils.data import DataLoader, WeightedRandomSampler
from metrics import get_partial_auc, plot_precision_recall_curve, plot_roc_with_partial_auc

BATCH_SIZE = 128

all_preds = []
all_labels = []
for folder in range(1, 6):
    train_dataset = ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=folder, return_tensors=True, balance=False)
    class_weights = len(train_dataset) / torch.bincount(train_dataset.labels)
    sampler = WeightedRandomSampler(
        weights=[class_weights[l.item()] for l in train_dataset.labels],
        num_samples=len(train_dataset),
        replacement=True
    )
    train_dataloader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=1,
                                sampler=sampler)

    test_dataloader = DataLoader(ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=folder, train=False, return_tensors=True), 
                                batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=1)

    best_features = {}
    numerical_features = []
    categorical_features = []
    with open('nb_best_features_fix.json', 'r') as f:
        best_features = json.load(f)
        numerical_features = [train_dataset.get_indexes(ft) for ft, keep in best_features.items() if keep and ft in ISIC2024.NUMERICAL_FEATURES]
        if len(numerical_features) > 0:
            numerical_features = np.concatenate(numerical_features)
        else:
            numerical_features = np.array([])
        categorical_features = [train_dataset.get_indexes(ft) for ft, keep in best_features.items() if keep and ft in ISIC2024.RAW_CATEGORICAL_FEATURES]
        if len(categorical_features) > 0:
            categorical_features = np.concatenate(categorical_features)
        else:
            categorical_features = np.array([])

    model = NaiveBayesEnsemble(n_estimators=32, sampling_strategy=0.01, random_state=42)
    model.fit(train_dataset.metadata, train_dataset.lbs, 
            categorical_features, numerical_features,
            weights=class_weights)

    for batch, labels in test_dataloader:
        all_preds.append(F.softmax(model(batch), dim=1)[:, 1])
        all_labels.append(labels)

preds = torch.cat(all_preds)
labels = torch.cat(all_labels)
print({
    'pAUC': get_partial_auc(preds.numpy(), labels.numpy(), min_tpr=0.80),
})

plot_roc_with_partial_auc(preds.numpy(), labels.numpy(), min_tpr=0.80, save_path='./roc_nb.png')
plot_precision_recall_curve(preds.numpy(), labels.numpy(), save_path='./pr_nb.png', min_recall=0.8)

