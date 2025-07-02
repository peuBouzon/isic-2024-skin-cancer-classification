from dataset import ISIC2024
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn
from torch.optim import Adam
from torchmetrics.functional import accuracy, recall, precision
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

BATCH_SIZE = 128
FOLDER = 1
EPOCHS = 200

def partial_auc(y_hat, y_true, min_tpr=0.80):
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

train_dataset = ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=FOLDER, return_tensors=True, balance=False)
class_weights = len(train_dataset) / torch.bincount(train_dataset.labels)
sampler = WeightedRandomSampler(
    weights=[class_weights[l.item()] for l in train_dataset.labels],
    num_samples=len(train_dataset),
    replacement=True
)
train_dataloader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=8, pin_memory=True,
                            sampler=sampler,
                            persistent_workers=True,
                            prefetch_factor=4)

test_dataloader = DataLoader(ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=FOLDER, train=False, return_tensors=True), 
                            batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, 
                            persistent_workers=True, prefetch_factor=2)

model = nn.Sequential(nn.LazyLinear(64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 2)) 
batch, labels = next(iter(train_dataloader))
loss_function = nn.CrossEntropyLoss(class_weights)
opt = Adam(model.parameters(), lr=1e-5)
lr_scheduler = ReduceLROnPlateau(opt, 'min', patience=10, factor=0.1, min_lr=1e-8)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

n_batches_train = len(train_dataloader)
n_batches_val = len(test_dataloader)
progress_bar = tqdm(range(EPOCHS))
for epochs in progress_bar:
    model.train()
    loss_epoch = 0
    for batch, labels in train_dataloader:
        batch.to(device) 
        labels.to(device)
        loss = loss_function(model(batch), labels)
        loss_epoch += loss.item()
        model.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        loss = 0
        for batch, labels in test_dataloader:
            all_preds.append(F.softmax(model(batch), dim=1).argmax(dim=1))
            all_labels.append(labels)
            loss += loss_function(model(batch), labels)
        
        #lr_scheduler.step(loss)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        progress_bar.set_postfix({
            'loss': (loss / n_batches_val).item(),
            'acc': accuracy(preds, labels, task='binary').item(),
            'recall': recall(preds, labels, task='binary').item(),
            'precision': precision(preds, labels, task='binary').item(),
            'pAUC': partial_auc(preds.numpy(), labels.numpy(), min_tpr=0.80),
            'lr': opt.param_groups[0]['lr']
            })
