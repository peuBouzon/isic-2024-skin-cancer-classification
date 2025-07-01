from dataset import ISIC2024
from torch.utils.data import DataLoader

BATCH_SIZE = 64
train_dataloader = DataLoader(ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=1), batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, 
                              persistent_workers=True, prefetch_factor=4)

test_dataloader = DataLoader(ISIC2024('./data/isic-2024-one-hot-encoded.csv', folder=1, train=False), batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, 
                              persistent_workers=True, prefetch_factor=4)

print(len(train_dataloader) * 64, len(test_dataloader) * 64)