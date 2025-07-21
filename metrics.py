import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

def get_partial_auc_scorer(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    return get_partial_auc(y_hat, y_true)

def get_partial_auc(y_hat, y_true, min_tpr=0.80):
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def get_precision_at_recall_95(probs, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    
    target_idxs = np.where(recall >= 0.95)[0]
    
    if len(target_idxs) > 0:
        target_idx = target_idxs[np.argmax(precision[target_idxs])]
        return precision[target_idx]
    else:
        raise ValueError()

def plot_precision_recall_curves(methods_names, 
                                methods_probs, 
                                methods_labels, 
                                save_path, 
                                min_tpr=0.8):
    
    plt.figure(figsize=(10, 8))
    colors = [['r', 'lightcoral'], ['g', 'mediumseagreen'], ['b', 'skyblue']]
    for i, (method_name, probs, y_true, (color, color_fill_between)) in enumerate(zip(methods_names, methods_probs, methods_labels, colors)):
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        average_precision = average_precision_score(y_true, probs)
        
        plt.fill_between(recall, precision, 
                color=color_fill_between, alpha=0.5, step='post')
        
        plt.step(recall, precision, color=color, where='post', lw=2, 
                label=f'Curva PR {method_name}\n(AP = {average_precision:.4f})')
        
        target_idxs = np.where(recall >= 0.95)[0]
        if len(target_idxs) > 0:
            target_idx = target_idxs[np.argmax(precision[target_idxs])]
            target_threshold = thresholds[target_idx]
            target_precision = precision[target_idx]
            target_recall_actual = recall[target_idx]
        plt.plot(target_recall_actual, target_precision, f'{color}o', markersize=8, 
                label=f'P={target_precision:.4f}, TPR={target_recall_actual:.4f}\n(Limiar = {target_threshold:.4f})')
        
    # Format the plot
    fontsize = 16
    plt.xlim([0.8, 1.0])
    plt.ylim([0.0, 0.05])
    plt.xlabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=fontsize)
    plt.ylabel('Precisão (P)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="best", fontsize=13)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
def plot_rocs_with_partial_aucs(methods_names, 
                                methods_probs, 
                                methods_labels, 
                                save_path, 
                                min_tpr=0.8):
    
    plt.figure(figsize=(10, 8))
    colors = [['r', 'lightcoral'], ['g', 'mediumseagreen'], ['b', 'skyblue']]
    for i, (method_name, probs, labels, (color, color_fill_between)) in enumerate(zip(methods_names, methods_probs, methods_labels, colors)):
        if i == 0:
            # Plot a linha de referencia de classificador aleatório
            plt.plot([0, 1], [0, 1], 'k--', lw=1)

            # Plota uma linha com TPR == min_tpr 
            plt.axhline(y=min_tpr, color='r', linestyle='--', 
                    label=f'Limiar de TPR mínimo = {min_tpr}')
            
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        #plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC {method_name}\n(pAUC @ 80% TPR = {get_partial_auc(probs, labels):.4f})')
        
        # Destaca a região com TPR >= min_tpr
        idx = np.where(tpr >= min_tpr)[0]
        if len(idx) > 0:
            pAUC = get_partial_auc(probs, labels, min_tpr=min_tpr)
            plt.fill_between(fpr[idx], tpr[idx], min_tpr, color=color_fill_between, alpha=0.5)
        
        
        # Marca um ponto referente ao threshold em que o modelo tem pelo menos 0.95 de TPR
        threshold_tpr = 0.95
        target_tpr_idx = np.argmin(np.abs(tpr - threshold_tpr))
        while (target_tpr_idx < len(tpr) - 1 and tpr[target_tpr_idx] < threshold_tpr):
            target_tpr_idx += 1
        target_threshold = thresholds[target_tpr_idx]
        target_fpr = fpr[target_tpr_idx]
        target_tpr = tpr[target_tpr_idx]
        plt.plot(target_fpr, target_tpr, f'{color}o', markersize=8, 
                label=f'TPR={target_tpr:.4f}, FPR={target_fpr:.4f}\n(Limiar = {target_threshold:.4f})')
    
    fontsize = 16
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=fontsize)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    preds = [0.1, 0.4, 0.35, 0.8, 0.7]
    labels = [0, 0, 1, 1, 1]
    preds = np.array(preds)
    labels = np.array(labels)
    print("Partial AUC:", get_partial_auc(preds, labels, min_tpr=0.8))