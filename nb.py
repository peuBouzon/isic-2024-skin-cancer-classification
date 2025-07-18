import json
import torch
import numpy as np
import torch.nn.functional as F

from dataset import ISIC2024
from torch.utils.data import DataLoader
from naivebayes import NaiveBayesEnsemble
from metrics import get_partial_auc, plot_precision_recall_curve, plot_roc_with_partial_auc

from sacred import Experiment
from sacred.observers import FileStorageObserver
from datetime import datetime
from pathlib import Path

ex = Experiment('nb_experiment')

@ex.config
def cfg():
	sampling_ratio = 0.01
	random_state = 42
	n_classifiers = 32
	data_path = './data/isic-2024-one-hot-encoded.csv'
	best_features_file = 'nb_best_features.json'
	batch_size = 128
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_folder = Path('results') / 'naivebayes' / timestamp
	ex.observers.append(FileStorageObserver(save_folder))

@ex.automain
def run(sampling_ratio, random_state, n_classifiers, batch_size, data_path, best_features_file, save_folder):
	all_preds = []
	all_labels = []
	for folder in range(1, 6):
		train_dataset = ISIC2024(data_path, folder=folder, return_tensors=True, balance=False)
		class_weights = len(train_dataset) / torch.bincount(train_dataset.labels)

		test_dataloader = DataLoader(ISIC2024(data_path, folder=folder, train=False, return_tensors=True), 
									batch_size=batch_size, shuffle=False, 
									num_workers=1)

		best_features = {}
		numerical_features = []
		categorical_features = []
		with open(best_features_file, 'r') as f:
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

		model = NaiveBayesEnsemble(n_estimators=n_classifiers, sampling_strategy=sampling_ratio, random_state=random_state)
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

	plot_roc_with_partial_auc(preds.numpy(), labels.numpy(), min_tpr=0.80, save_path=save_folder / './roc_curve.png')
	plot_precision_recall_curve(preds.numpy(), labels.numpy(), save_path=save_folder / './precision_recall_curve.png', min_recall=0.8)

