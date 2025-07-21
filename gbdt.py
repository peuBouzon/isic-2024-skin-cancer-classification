import json

import pandas as pd
import config
import numpy as np
import polars as pl
import lightgbm as lgb

from pathlib import Path
from dataset import ISIC2024
from sacred import Experiment
from datetime import datetime
from imblearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sacred.observers import FileStorageObserver
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold
from metrics import get_partial_auc_scorer, get_partial_auc

ex = Experiment('gbdt_experiment')

@ex.config
def cfg():
	sampling_ratio = 0.01
	random_state = 42
	n_classifiers = 32
	data_path = config.RAW_METADATA_PATH
	best_params_path = config.GBDT_BEST_HYPERPARAMETERS_FILE
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	save_folder = Path('results') / 'gbdt' / timestamp
	ex.observers.append(FileStorageObserver(save_folder))

@ex.automain
def run(sampling_ratio, random_state, n_classifiers, data_path, best_params_path, save_folder):
	best_params = {}
	with open(best_params_path, 'r') as f:
		best_params = json.load(f)

	with open(save_folder / 'best_hyperparameters.json', 'w') as f:
		json.dump(best_params, f, indent=4)

	estimator = VotingClassifier([
		(f'lgb{i}', Pipeline([
			('sampler', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=7+i*5+random_state)),
			('classifier', lgb.LGBMClassifier(**best_params, random_state=7+i*5+random_state)),
		])) for i in range(n_classifiers)
	], voting='soft')


	df = (pl.read_csv(data_path)
		.with_columns(pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64))
		.with_columns(pl.col(ISIC2024.RAW_CATEGORICAL_FEATURES).cast(pl.Categorical),
		)
		.to_pandas())

	X = df[ISIC2024.RAW_CATEGORICAL_FEATURES + ISIC2024.NUMERICAL_FEATURES]
	y = df[ISIC2024.TARGET_COLUMN]
	groups = df[ISIC2024.PATIENT_ID]
	cv = StratifiedGroupKFold(5, shuffle=True, random_state=random_state)

	# Define multiple scoring metrics
	scoring = {
		'pAUC_80': get_partial_auc_scorer,
	}

	# Use cross_validate instead of cross_val_score
	cv_results = cross_validate(
		estimator=estimator, 
		X=X, y=y,
		cv=cv, 
		groups=groups,
		scoring=scoring,
		return_train_score=False,
		return_estimator=True,
		return_indices=True
	)

	print("Results for each fold:")
	print("-" * 50)
	for fold in range(len(cv_results['test_pAUC_80'])):
		print(f"Fold {fold + 1}:")
		print(f"  pAUC: {cv_results['test_pAUC_80'][fold]:.4f}")

	print("Average scores across all folds:")
	print(f"pAUC: {np.mean(cv_results['test_pAUC_80']):.4f} ± {np.std(cv_results['test_pAUC_80']):.4f}")

	y_pred, y_true = [], []
	for folder in range(5):
		val_indexes = ~X.index.isin(cv_results['indices']['train'][folder])
		preds = cv_results['estimator'][folder].predict_proba(X[val_indexes])[:, 1]
		labels = y[val_indexes]
		y_pred.extend(preds)
		y_true.extend(labels)

		predictions_df = pd.DataFrame.from_dict({
			'preds': preds,
			'labels': labels,
		})
		predictions_file = save_folder / f'folder_{folder + 1}' / 'predictions.csv'
		predictions_file.parent.mkdir(exist_ok=True)
		predictions_df.to_csv(predictions_file, index=False)

	y_pred, y_true = np.array(y_pred), np.array(y_true)

	print(f"Partial AUC (80% TPR): {get_partial_auc(y_pred, y_true, min_tpr=0.80):.4f}")