import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from dataset import ISIC2024
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional import accuracy, recall, precision
from metrics import get_partial_auc, plot_precision_recall_curve, plot_roc_with_partial_auc

BATCH_SIZE = 512
EPOCHS = 512

all_preds = []
all_labels = []
for folder in range(1, 6):
    train_dataset = ISIC2024('./data/isic-2024-one-hot-encoded.csv', 
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
                                batch_size=BATCH_SIZE,
                                num_workers=8, pin_memory=True,
                                sampler=sampler,
                                persistent_workers=True,
                                prefetch_factor=4)

    val_dataset = ISIC2024('./data/isic-2024-one-hot-encoded.csv', 
                        folder=folder,
                        train=False,
                        return_tensors=True,
                        fillna=True,
                        imputer=train_dataset.get_imputer())

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=4, pin_memory=True, 
                                persistent_workers=True, prefetch_factor=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nn.Sequential(nn.LazyLinear(128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(64, 2)).to(device)

    loss_function = nn.CrossEntropyLoss(class_weights.to(device))

    opt = Adam(model.parameters(), lr=1e-5)

    progress_bar = tqdm(range(EPOCHS))

    best_metric = 0
    min_loss = float('inf')
    for epochs in progress_bar:
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
            all_preds = []
            all_labels = []
            loss = 0
            for batch, labels in val_dataloader:
                batch = batch.to(device) 
                labels = labels.to(device)
                preds = model(batch)
                all_preds.append(F.softmax(preds, dim=1)[:, 1])
                all_labels.append(labels)
                loss += loss_function(preds, labels)
            
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)

            pAUC = get_partial_auc(preds.cpu().numpy(), labels.cpu().numpy(), min_tpr=0.80)

            if best_metric < pAUC:
                best_metric = pAUC

            if loss < min_loss:
                min_loss = loss
                torch.save(model.state_dict(), f'./best-checkpoint.pth')

            progress_bar.set_postfix({
                'loss': (loss / len(val_dataloader)).item(),
                'acc': accuracy(preds, labels, task='binary').item(),
                'recall': recall(preds, labels, task='binary').item(),
                'precision': precision(preds, labels, task='binary').item(),
                'pAUC': pAUC,
                })

    model.load_state_dict(torch.load('./best-checkpoint.pth'))
    model.eval()

    with torch.no_grad():
        loss = 0
        for batch, labels in val_dataloader:
            batch = batch.to(device) 
            labels = labels.to(device)
            preds = model(batch)
            all_preds.append(F.softmax(preds, dim=1)[:, 1])
            all_labels.append(labels)

preds = torch.cat(all_preds)
labels = torch.cat(all_labels)

pAUC = get_partial_auc(preds.cpu().numpy(), labels.cpu().numpy(), min_tpr=0.80)
print(f'pAUC final: {pAUC}')
plot_precision_recall_curve(preds.cpu().numpy(), labels.cpu().numpy(), save_path='./mlp-precision_recall_curve.png')
plot_roc_with_partial_auc(preds.cpu().numpy(), labels.cpu().numpy(), save_path='./mlp-roc_curve.png', min_tpr=0.80)
