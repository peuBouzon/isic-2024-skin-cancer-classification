import torch
import config
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from dataset import ISIC2024
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional import accuracy, recall, precision
from metrics import get_partial_auc, plot_precision_recall_curve, plot_roc_with_partial_auc

from sacred import Experiment
from sacred.observers import FileStorageObserver
from datetime import datetime
from pathlib import Path

ex = Experiment('mlp_experiment')

@ex.config
def cfg():
	data_path = config.ONE_HOT_ENCODED_PATH
	batch_size = 256
	epochs = 300
	warm_up_epochs = 150
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_folder = Path('results') / 'mlp' / timestamp
	ex.observers.append(FileStorageObserver(save_folder))

@ex.automain
def run(batch_size, epochs, warm_up_epochs, data_path, save_folder):

	assert epochs > warm_up_epochs, "O número de épocas deve ser maior que o número de épocas de aquecimento."

	all_preds = []
	all_labels = []
	metrics_per_folder = {
		'folder': [],
		'pAUC': [],
	}
	for folder in range(1, 6):
		train_dataset = ISIC2024(data_path, 
								folder=folder,
								return_tensors=True,
								balance=True,
								fillna=True)

		class_weights = len(train_dataset) / torch.bincount(train_dataset.labels)

		sampler = WeightedRandomSampler(
			weights=[class_weights[l.item()] for l in train_dataset.labels],
			num_samples=len(train_dataset),
			replacement=True
		)

		train_dataloader = DataLoader(train_dataset,
									batch_size=batch_size,
									num_workers=1,
									sampler=sampler,)

		val_dataset = ISIC2024(data_path, 
							folder=folder,
							train=False,
							return_tensors=True,
							fillna=True,
							imputer=train_dataset.get_imputer())

		val_dataloader = DataLoader(val_dataset, 
									batch_size=batch_size, shuffle=False, 
									num_workers=1)

		device = 'cuda' if torch.cuda.is_available() else 'cpu'

		model = nn.Sequential(nn.LazyLinear(128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
							nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
							nn.Linear(64, 2)).to(device)

		loss_function = nn.CrossEntropyLoss(class_weights.to(device))

		opt = Adam(model.parameters(), lr=1e-5)

		progress_bar = tqdm(range(epochs))

		best_metric = 0
		min_loss = float('inf')
		checkpoint_path = save_folder / f'folder_{folder}' / f'best-checkpoint.pth'
		checkpoint_path.parent.mkdir(exist_ok=True)
	
		for epoch in progress_bar:
			model.train()
			loss_epoch = 0
			for batch, labels in train_dataloader:
				batch = batch.to(device) 
				labels = labels.to(device)
				loss = loss_function(model(batch), labels)
				loss_epoch += loss.item()
				model.zero_grad()
				loss.backward()
				opt.step()

			model.eval()
			with torch.no_grad():
				folder_preds = []
				folder_labels = []
				loss_epoch = 0
				for batch, labels in val_dataloader:
					batch = batch.to(device) 
					labels = labels.to(device)
					preds = model(batch)
					folder_preds.append(F.softmax(preds, dim=1)[:, 1])
					folder_labels.append(labels)
					loss_epoch += loss_function(preds, labels)
				
				preds = torch.cat(folder_preds)
				labels = torch.cat(folder_labels)

				pAUC = get_partial_auc(preds.cpu().numpy(), labels.cpu().numpy(), min_tpr=0.80)

				if best_metric < pAUC:
					best_metric = pAUC

				if loss_epoch < min_loss and epoch >= warm_up_epochs:
					min_loss = loss
					torch.save(model.state_dict(), checkpoint_path)

				progress_bar.set_postfix({
					'loss': (loss_epoch / len(val_dataloader)).item(),
					'acc': accuracy(preds, labels, task='binary').item(),
					'recall': recall(preds, labels, task='binary').item(),
					'precision': precision(preds, labels, task='binary').item(),
					'pAUC': pAUC,
					})

		model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
		model.eval()

		with torch.no_grad():
			folder_preds = []
			folder_labels = []
			for batch, labels in val_dataloader:
				folder_labels.append(labels.to(device))
				folder_preds.append(F.softmax(model(batch.to(device)), dim=1)[:, 1])

			preds = torch.cat(folder_preds)
			labels = torch.cat(folder_labels)
			all_preds.append(preds)
			all_labels.append(labels)

			pAUC = get_partial_auc(preds.cpu().numpy(), labels.cpu().numpy(), min_tpr=0.80)
			print(pAUC)
			metrics_per_folder['folder'].append(folder)
			metrics_per_folder['pAUC'].append(pAUC)
			predictions_df = pd.DataFrame.from_dict({
				'preds': preds.cpu().numpy(),
				'labels': labels.cpu().numpy(),
			})
			predictions_df.to_csv(save_folder / f'folder_{folder}' / 'predictions.csv', index=False)


		metrics_df = pd.DataFrame.from_dict(metrics_per_folder)
		metrics_df.to_csv(save_folder / f'folder_{folder}' / 'metrics.csv', index=False)

	all_preds = torch.cat(all_preds).cpu().numpy()
	all_labels = torch.cat(all_labels).cpu().numpy()
	pAUC = get_partial_auc(np.array(all_preds), np.array(all_labels), min_tpr=0.80)
	print(f'pAUC final: {pAUC}')
	plot_precision_recall_curve(np.array(all_preds), np.array(all_labels), save_path= save_folder / './precision_recall_curve.png')
	plot_roc_with_partial_auc(np.array(all_preds), np.array(all_labels), save_path= save_folder / './roc_curve.png', min_tpr=0.80)
