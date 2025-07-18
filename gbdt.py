import json
from sklearn.ensemble import VotingClassifier
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
from imblearn.pipeline import Pipeline
from dataset import ISIC2024
from sklearn.model_selection import cross_validate
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from metrics import get_partial_auc_scorer, plot_precision_recall_curve, plot_roc_with_partial_auc, get_partial_auc

SAMPLING_RATIO = 0.01
RANDOM_STATE = 42
N_CLASSIFIERS = 1

best_params = {}
with open('best_hyperparameters.json', 'r') as f:
  best_params = json.load(f)

estimator = VotingClassifier([
    (f'lgb{i}', Pipeline([
        ('sampler', RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=7+i*5)),
        ('classifier', lgb.LGBMClassifier(**best_params, random_state=7+i*5)),
    ])) for i in range(N_CLASSIFIERS)
], voting='soft')


df = (pl.read_csv('./train-metadata.csv')
    .with_columns(pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64))
    .with_columns(pl.col(ISIC2024.RAW_CATEGORICAL_FEATURES).cast(pl.Categorical),
      )
      .to_pandas())

X = df[ISIC2024.RAW_CATEGORICAL_FEATURES + ISIC2024.NUMERICAL_FEATURES]
y = df[ISIC2024.TARGET_COLUMN]
groups = df[ISIC2024.PATIENT_ID]
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

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
  y_pred.extend(cv_results['estimator'][folder].predict_proba(X[val_indexes])[:, 1])
  y_true.extend(y[val_indexes])

y_pred, y_true = np.array(y_pred), np.array(y_true)

print(f"Partial AUC (80% TPR): {get_partial_auc(y_pred, y_true, min_tpr=0.80):.4f}")
plot_precision_recall_curve(y_pred, y_true)
plot_roc_with_partial_auc(y_pred, y_true, min_tpr=0.80)