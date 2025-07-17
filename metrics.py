import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

def get_partial_auc(y_hat, y_true, min_tpr=0.80):
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def plot_precision_recall_curve(probs, y_true, min_recall=0.8):
    
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    average_precision = average_precision_score(y_true, probs)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', where='post', lw=2, 
             label=f'Curva de Precisão-Sensibilidade (AP = {average_precision:.4f})')
    
    high_recall_idxs = np.where(recall >= min_recall)[0]
    print(high_recall_idxs)
    if len(high_recall_idxs) > 0:
        high_recall_auc = auc(recall[high_recall_idxs], precision[high_recall_idxs])
        plt.fill_between(recall[high_recall_idxs], precision[high_recall_idxs], 
                color='skyblue', alpha=0.5, step='post',
                label=f'Área com Sensibilidade >= {min_recall}, AUC ≈ {high_recall_auc:.4f})')
        plt.axvline(x=min_recall, color='r', linestyle='--', 
                   label=f'Limitar de Sensibilidade = {min_recall}')
    
    target_idxs = np.where(recall >= 0.95)[0]
    if len(target_idxs) > 0:
        target_idx = target_idxs[np.argmax(precision[target_idxs])]
        target_threshold = thresholds[target_idx]
        target_precision = precision[target_idx]
        target_recall_actual = recall[target_idx]
    plt.plot(target_recall_actual, target_precision, 'ro', markersize=8, 
            label=f'Limitar = {target_threshold:.4f} (P={target_precision:.4f}, S={target_recall_actual:.4f})')
    
    # Format the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Sensibilidade (S)')
    plt.ylabel('Precisão (P)')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')


def plot_roc_with_partial_auc(probs, y_true, min_tpr=0.8):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
   
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    
    # Destaca a região com TPR >= min_tpr
    idx = np.where(tpr >= min_tpr)[0]
    if len(idx) > 0:
        pAUC = get_partial_auc(probs, y_true, min_tpr=min_tpr)
        plt.fill_between(fpr[idx], tpr[idx], min_tpr, color='skyblue', alpha=0.5,
                        label=f'AUC parcial (TPR ≥ {min_tpr}) = {pAUC:.4f}')
    
    # Plot a linha de referencia de classificador aleatório
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # Plota uma linha com TPR == min_tpr 
    plt.axhline(y=min_tpr, color='r', linestyle='--', 
               label=f'Limitar de TPR mínimo = {min_tpr}')
    
    # Marca um ponto referente ao threshold em que o modelo tem pelo menos 0.95 de TPR
    threshold_tpr = 0.95
    target_tpr_idx = np.argmin(np.abs(tpr - threshold_tpr))
    while (target_tpr_idx < len(tpr) - 1 and tpr[target_tpr_idx] < threshold_tpr):
        target_tpr_idx += 1
    target_threshold = thresholds[target_tpr_idx]
    target_fpr = fpr[target_tpr_idx]
    target_tpr = tpr[target_tpr_idx]
    plt.plot(target_fpr, target_tpr, 'ro', markersize=8, 
            label=f'Limitar = {target_threshold:.4f} (TPR={target_tpr:.4f}, FPR={target_fpr:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve_with_partial_auc.png', dpi=300, bbox_inches='tight')
    

if __name__ == '__main__':
    preds = [0.1, 0.4, 0.35, 0.8, 0.7]
    labels = [0, 0, 1, 1, 1]
    preds = np.array(preds)
    labels = np.array(labels)
    print("Partial AUC:", get_partial_auc(preds, labels, min_tpr=0.8))
    plot_precision_recall_curve(preds, labels, min_recall=0.8)
    plot_roc_with_partial_auc(preds, labels, min_tpr=0.8)