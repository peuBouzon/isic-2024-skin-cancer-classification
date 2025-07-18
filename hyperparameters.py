import json
import optuna
import config
import numpy as np
import polars as pl
from dataset import ISIC2024
from functools import partial
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from metrics import get_partial_auc_scorer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

def objective(trial, X, y, groups):
	param = {
		'objective': 'binary',
		'verbosity': -1,
		'n_jobs': -1,
		'n_estimators': 200,
		'boosting_type': 'gbdt',
		'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
		'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
		'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
		'max_depth': trial.suggest_int('max_depth', 3, 12),
		'num_leaves': trial.suggest_int('num_leaves', 31, 255),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
		'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
		'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
		'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
		'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
		'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
	}

	pipeline = Pipeline([
		('sampler', RandomUnderSampler(sampling_strategy=0.01, random_state=42)),
		('classifier', LGBMClassifier(**param, random_state=42))
	])

	scoring = {
		'pAUC_80': get_partial_auc_scorer,
	}

	cv_results = cross_validate(
		estimator=pipeline, 
		X=X, y=y,
		cv=cv, 
		groups=groups,
		scoring=scoring,
		return_train_score=False,
	)

	return np.mean(cv_results['test_pAUC_80'])

if __name__ == "__main__":
	df = (pl.read_csv(config.RAW_METADATA_PATH)
			.with_columns(pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64))
			.with_columns(pl.col(ISIC2024.RAW_CATEGORICAL_FEATURES).cast(pl.Categorical),
        )
        .to_pandas())

	X = df[ISIC2024.RAW_CATEGORICAL_FEATURES + ISIC2024.NUMERICAL_FEATURES]
	y = df[ISIC2024.TARGET_COLUMN]
	groups = df[ISIC2024.PATIENT_ID]
	cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

	study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
	study.optimize(partial(objective, X=X, y=y, groups=groups), n_trials=100, n_jobs=-1)

	best_params = {
		'objective': 'binary',
		'verbosity': -1,
		'n_jobs': 2,
		'n_estimators': 200,
		'boosting_type': 'gbdt',
		**study.best_params,
	}

	print(f'Salvando {config.GBDT_BEST_HYPERPARAMETERS_FILE}...')
	with open(config.GBDT_BEST_HYPERPARAMETERS_FILE, 'w') as f:
		json.dump(best_params, f, indent=4)
