from pathlib import Path
import numpy as np
import pandas as pd
from metrics import get_partial_auc, plot_precision_recall_curves, plot_rocs_with_partial_aucs, get_precision_at_recall_95
import matplotlib.pyplot as plt
from stats import statistical_test
from config import EXPERIMENT_DIRS_BY_METHOD
colors = iter([['r', 'lightcoral'], ['g', 'mediumseagreen'], ['b', 'skyblue']])
all_preds = []
all_labels = []
methods_names = []
names_map = {
  'naivebayes': 'Naive Bayes',
  'mlp' : 'MLP',
  'gbdt': 'GBDT'
}

# experiment => RESULTS_FOLDER / {method} / {experiment_dir}
RESULTS_FOLDER = Path('results')

pAUCs = []
precisions95recall = []
for i, (method_name, experiment_dirs) in enumerate(EXPERIMENT_DIRS_BY_METHOD.items()):
  methods_names.append(method_name)
  method_pAUCs = []
  pAUCs.append(method_pAUCs)
  method_pr95recall = []
  precisions95recall.append(method_pr95recall)
  color = next(colors)
  preds = []
  labels = []
  for timestamp_dir in experiment_dirs:
    timestamp_dir_path = (RESULTS_FOLDER / method_name / timestamp_dir)
    for folder_path in timestamp_dir_path.iterdir():
      if folder_path.stem.startswith('folder'):
        df = pd.read_csv(folder_path / 'predictions.csv')
        method_pAUCs.append(get_partial_auc(df['preds'], df['labels']))
        method_pr95recall.append(get_precision_at_recall_95(df['preds'], df['labels']))
        preds.extend(df['preds'])
        labels.extend(df['labels'])
  preds = np.array(preds)
  labels = np.array(labels)
  all_preds.append(preds)
  all_labels.append(labels)
  ax = plt.subplot(111)

pAUCs = np.array(pAUCs)
for method, pAUC_mean, pAUC_std in zip(methods_names, np.mean(pAUCs, axis=1), np.std(pAUCs, axis=1)):
  print(f'pAUC for {method}: {pAUC_mean} +- {pAUC_std}')

precisions95recall = np.array(precisions95recall)
for method, pr_mean, pr_std in zip(methods_names, np.mean(precisions95recall, axis=1), np.std(precisions95recall, axis=1)):
  print(f'Precision @ TPR = 95% for {method}: {pr_mean} +- {pr_std}')

statistical_test(pAUCs, methods_names, pv_wilcoxon=0.05)
plot_rocs_with_partial_aucs(methods_names, all_preds, all_labels, RESULTS_FOLDER / 'roc.png')
plot_precision_recall_curves(methods_names, all_preds, all_labels, RESULTS_FOLDER / 'pr.png')